"""
train.py
========
多模态大模型联合微调脚本 (通用版，支持 CWRU / XJTU 等所有由 generate_dataset 产出的 SFT 数据)
"""
import os
import sys
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

# 获取项目根目录并加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multimodal_qwen import BearingMultimodalQwen

QWEN_PATH    = os.path.join(PROJECT_ROOT, "qwen_weights")
DATA_DIR     = PROJECT_ROOT
OUTPUT_DIR   = os.path.join(DATA_DIR, "bearllm_weights", "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_PATH    = os.path.join(DATA_DIR, "data", "train_sft.json")
PT_PATH      = os.path.join(DATA_DIR, "data", "train_features.pt")

# 超参数
BATCH_SIZE           = 1
GRAD_ACCUM_STEPS     = 8
NUM_EPOCHS           = 3
LEARNING_RATE        = 2e-4
MAX_SEQ_LEN          = 512
WARMUP_RATIO         = 0.05
SAVE_STEPS           = 50

# LoRA
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LORA_TARGET     = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE_INDEX    = -100

class MultimodalSFTDataset(Dataset):
    SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "你是一名专业的轴承故障诊断专家。结合输入的底层振动信号语义特征和时空图谱分析，"
        "请给出严谨的推理过程与最终诊断结论。\n<|im_end|>\n"
    )

    def __init__(self, json_path, pt_path, tokenizer, max_len=MAX_SEQ_LEN):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        self.features = torch.load(pt_path, map_location='cpu')
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        feat = self.features[idx]

        user_text = (
            f"<|im_start|>user\n"
            f"{rec['instruction']}\n\n"
            f"{rec['input']}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = self.SYSTEM_PROMPT + user_text

        prompt_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(rec['output'] + "\n<|im_end|>", add_special_tokens=False).input_ids

        max_prompt_len = self.max_len - min(len(response_ids), 64) - 2
        prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + response_ids
        input_ids = input_ids[:self.max_len]

        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        labels = labels[:self.max_len]

        pad_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len

        return {
            'deep_feature': feat,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

def collate_fn(batch):
    return {
        'deep_feature'  : torch.stack([b['deep_feature'] for b in batch]),
        'input_ids'     : torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels'        : torch.stack([b['labels'] for b in batch]),
    }

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

def save_standard_checkpoint(wrapper, optimizer, scheduler, epoch, step, loss, path):
    os.makedirs(path, exist_ok=True)
    wrapper.qwen.save_pretrained(os.path.join(path, "lora_adapter"))
    torch.save(wrapper.alignment_layer.state_dict(), os.path.join(path, "alignment_layer.pt"))
    print(f"  → Checkpoint 已标准化保存: {path}")

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 训练设备: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    dataset = MultimodalSFTDataset(JSON_PATH, PT_PATH, tokenizer, max_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    wrapper = BearingMultimodalQwen(qwen_path=QWEN_PATH, freeze_llm=True, num_vib_tokens=8)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
    )
    wrapper.qwen = get_peft_model(wrapper.qwen, lora_config)
    wrapper.qwen.print_trainable_parameters()

    for param in wrapper.alignment_layer.parameters():
        param.requires_grad = True
    ALIGN_DEVICE = next(wrapper.qwen.parameters()).device   
    wrapper.alignment_layer = wrapper.alignment_layer.to(ALIGN_DEVICE)

    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = (len(loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_loss = float('inf')
    global_step = 0
    accum_loss = 0.0

    print("\n🚀 开始联合微调...")
    for epoch in range(NUM_EPOCHS):
        wrapper.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, batch in pbar:
            outputs = wrapper(
                deep_feature=batch['deep_feature'].to(ALIGN_DEVICE),
                input_ids=batch['input_ids'].to(ALIGN_DEVICE),
                attention_mask=batch['attention_mask'].to(ALIGN_DEVICE),
                labels=batch['labels'].to(ALIGN_DEVICE),
            )

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            accum_loss += loss.item()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accum_loss

                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
                accum_loss = 0.0

                if global_step % SAVE_STEPS == 0:
                    ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_step{global_step}")
                    save_standard_checkpoint(wrapper, optimizer, scheduler, epoch, global_step, avg_loss, ckpt_path)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(OUTPUT_DIR, "best_checkpoint")
                    save_standard_checkpoint(wrapper, optimizer, scheduler, epoch, global_step, best_loss, best_path)

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\n[Epoch {epoch+1}] 平均 loss: {avg_epoch_loss:.4f} | 最优 loss: {best_loss:.4f}")

    final_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    save_standard_checkpoint(wrapper, optimizer, scheduler, NUM_EPOCHS, global_step, best_loss, final_path)
    print(f"\n✅ 训练大功告成！特征对齐网络与 LLM 知识引擎已深度绑定，请查看 {best_path}。")

if __name__ == "__main__":
    main()