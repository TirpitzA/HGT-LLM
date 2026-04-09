"""
evaluate_multimodal.py
======================
多模态模型评估脚本：通用版。读取 YAML 配置动态切换数据集，并评估多模态生成质量。
"""
import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 获取项目根目录并加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multimodal_qwen import AlignmentLayer

# ── 路径默认值 ──────────────────────────────────────────────────────────────
QWEN_PATH = "/root/autodl-tmp/XJTU_Multimodal_LLM/qwen_weights"
DATA_DIR  = "/root/autodl-tmp/XJTU_Multimodal_LLM"

class MultimodalEvaluator:
    SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "你是一名专业的轴承故障诊断专家。结合输入的底层振动信号语义特征和时空图谱分析，"
        "请给出严谨的推理过程与最终诊断结论。\n<|im_end|>\n"
    )

    def __init__(self, checkpoint_dir: str, qwen_path: str = QWEN_PATH, num_vib_tokens: int = 8):
        self.num_vib_tokens = num_vib_tokens
        lora_path  = os.path.join(checkpoint_dir, "lora_adapter")
        align_path = os.path.join(checkpoint_dir, "alignment_layer.pt")

        print("[EVAL] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        self.tokenizer.pad_token_id  = self.tokenizer.eos_token_id
        self.tokenizer.padding_side  = 'left'   

        print("[EVAL] 加载 Qwen 基座模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype  = torch.float16,
            device_map   = "auto",
            trust_remote_code = True,
        )

        print(f"[EVAL] 注入 LoRA 适配器: {lora_path}")
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()

        hidden_size = self.model.config.hidden_size
        self.device = next(self.model.parameters()).device

        self.alignment_layer = AlignmentLayer(
            input_dim = 64,
            hidden_dim = hidden_size,
            num_tokens = num_vib_tokens,
        ).to(self.device).to(torch.float16)

        print(f"[EVAL] 加载 AlignmentLayer 权重: {align_path}")
        self.alignment_layer.load_state_dict(
            torch.load(align_path, map_location=self.device)
        )
        self.alignment_layer.eval()
        print("[EVAL] 模型就绪✓")

    @torch.no_grad()
    def compute_ppl(self, deep_feature: torch.Tensor, instruction: str,
                    input_ctx: str, output_txt: str) -> float:
        """独立计算给定样本的条件语言模型困惑度 (PPL)，绝不污染模型状态"""
        self.model.eval()
        self.alignment_layer.eval()
        
        if deep_feature.dim() == 1:
            deep_feature = deep_feature.unsqueeze(0)
        deep_feature = deep_feature.to(self.device).to(torch.float16)

        user_text = (
            f"<|im_start|>user\n"
            f"{instruction}\n\n"
            f"{input_ctx}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_prompt = self.SYSTEM_PROMPT + user_text

        prompt_ids = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        target_ids = self.tokenizer(output_txt + "<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        seq_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(seq_ids)
        
        text_embeds = self.model.get_input_embeddings()(seq_ids)
        vib_embeds = self.alignment_layer(deep_feature)
        
        inputs_embeds = torch.cat([vib_embeds, text_embeds], dim=1)
        
        vib_mask = torch.ones((1, self.num_vib_tokens), dtype=attention_mask.dtype, device=self.device)
        extended_mask = torch.cat([vib_mask, attention_mask], dim=1)
        
        # We also need labels to use Qwen's built-in loss computation or compute manually
        vib_labels = torch.full((1, self.num_vib_tokens), -100, dtype=torch.long, device=self.device)
        prompt_labels = torch.full_like(prompt_ids, -100)
        extended_labels = torch.cat([vib_labels, prompt_labels, target_ids], dim=1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=extended_labels
        )
        
        loss = outputs.loss
        ppl = torch.exp(loss)
        return ppl.item()

    @torch.no_grad()
    def generate_one(self, deep_feature: torch.Tensor, instruction: str,
                     input_ctx: str, max_new_tokens: int = 256) -> str:
        
        if deep_feature.dim() == 1:
            deep_feature = deep_feature.unsqueeze(0)   # (1, 64)
        deep_feature = deep_feature.to(self.device).to(torch.float16)

        user_text = (
            f"<|im_start|>user\n"
            f"{instruction}\n\n"
            f"{input_ctx}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_prompt = self.SYSTEM_PROMPT + user_text

        enc = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids      = enc.input_ids         
        attention_mask = enc.attention_mask    

        embed_fn    = self.model.get_input_embeddings()
        text_embeds = embed_fn(input_ids)      
        vib_embeds  = self.alignment_layer(deep_feature)  

        inputs_embeds = torch.cat([vib_embeds, text_embeds], dim=1)
        vib_mask = torch.ones(1, self.num_vib_tokens, dtype=attention_mask.dtype, device=self.device)
        extended_mask = torch.cat([vib_mask, attention_mask], dim=1)

        # 严格保持 do_sample=False (Greedy Decoding)
        output_ids = self.model.generate(
            inputs_embeds    = inputs_embeds,
            attention_mask   = extended_mask,
            max_new_tokens   = max_new_tokens,
            do_sample        = False,
            pad_token_id     = self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_label(text: str, class_map: dict) -> Optional[str]:
    """从生成文本中提取分类标签"""
    for cls_id, cls_str in class_map.items():
        if cls_str in text:
            return cls_str
            
    for cls_id, cls_str in class_map.items():
        zh_part = cls_str.split(" (")[0]
        if zh_part in text:
            return cls_str
            
    return None

def evaluate(config_path: str, checkpoint_dir: str, json_path: str, pt_path: str,
             qwen_path: str = QWEN_PATH, max_new_tokens: int = 256):
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_name = config.get("dataset_name", "Unknown")
    class_map = config.get("labels", {})
    num_classes = config["model_params"]["num_classes"]

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    features = torch.load(pt_path, map_location='cpu')   
    assert len(records) == features.shape[0]

    evaluator = MultimodalEvaluator(checkpoint_dir, qwen_path)

    total   = len(records)
    correct = 0
    per_class_correct = {i: 0 for i in class_map.keys()}
    per_class_total   = {i: 0 for i in class_map.keys()}
    failures = []
    
    total_ppl = 0.0

    print(f"\n[EVAL] 开始评估 {total} 条 {dataset_name} 集样本...")
    for i in tqdm(range(total)):
        rec  = records[i]
        feat = features[i]   

        true_id  = rec['true_label']
        true_str = class_map[true_id]
        per_class_total[true_id] += 1

        # 计算 PPL
        ppl_score = evaluator.compute_ppl(
            deep_feature=feat,
            instruction=rec['instruction'],
            input_ctx=rec['input'],
            output_txt=rec['output']
        )
        total_ppl += ppl_score

        # 生成推断
        generated = evaluator.generate_one(
            deep_feature = feat,
            instruction  = rec['instruction'],
            input_ctx    = rec['input'],
            max_new_tokens = max_new_tokens,
        )

        pred_str = extract_label(generated, class_map)

        if pred_str == true_str:
            correct += 1
            per_class_correct[true_id] += 1
        else:
            failures.append({
                "idx": i, "true": true_str, "pred": pred_str,
                "generated": generated[:200]
            })

    accuracy = correct / total * 100
    avg_ppl = total_ppl / total

    print(f"\n{'='*70}")
    print(f"  {dataset_name} 多模态 LLM 轴承诊断评估结果 ({num_classes} 分类)")
    print(f"{'='*70}")
    print(f"  总样本数        : {total}")
    print(f"  正确预测        : {correct}")
    print(f"  诊断准确率      : {accuracy:.2f}%")
    print(f"  平均 PPL        : {avg_ppl:.4f}")
    print(f"{'='*70}")
    print("  Per-class 准确率：")
    for cls_id, cls_name in class_map.items():
        n  = per_class_total[cls_id]
        c  = per_class_correct[cls_id]
        pct = c / n * 100 if n > 0 else 0.0
        print(f"    [{cls_id}] {cls_name:<40} {c:>4}/{n:<4}  ({pct:.1f}%)")
    print(f"{'='*70}")

    if failures:
        print(f"\n  ⚠️  解析失败 / 错误预测：{len(failures)} 条 (前5条展示)")
        for f in failures[:5]:
            print(f"    idx={f['idx']} | true={f['true']} | pred={f['pred']}")
            print(f"    generated: {f['generated']}...")

    return accuracy, avg_ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态轴承诊断 LLM 评估脚本")
    # 这里为了向后兼容和简单化，我们可以提供 default 路径
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(DATA_DIR, "bearllm_weights/checkpoints/best_checkpoint"),
                        help="Checkpoint 目录（内含 lora_adapter/ + alignment_layer.pt）")
    parser.add_argument("--test_json", type=str,
                        default=os.path.join(DATA_DIR, "data", "test_sft.json"),
                        help="测试数据 JSON 路径")
    parser.add_argument("--test_pt",   type=str,
                        default=os.path.join(DATA_DIR, "data", "test_features.pt"),
                        help="测试 deep_feature .pt 路径")
    parser.add_argument("--qwen_path", type=str, default=QWEN_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    evaluate(
        config_path    = args.config,
        checkpoint_dir = args.checkpoint,
        json_path      = args.test_json,
        pt_path        = args.test_pt,
        qwen_path      = args.qwen_path,
        max_new_tokens = args.max_new_tokens,
    )