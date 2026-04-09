"""
evaluate_multimodal.py
======================
多模态模型评估脚本：适配 CWRU 10 分类任务。
"""
import os
import sys
import json
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

# ── 标签映射（CWRU 10分类） ─────────────────────────────────────────────────
CLASS_MAP_ZH = {
    0: "健康状态 (Healthy)",
    1: "内圈轻度故障_0.007 (Minor Inner Race)",
    2: "内圈中度故障_0.014 (Moderate Inner Race)",
    3: "内圈重度故障_0.021 (Severe Inner Race)",
    4: "滚珠轻度故障_0.007 (Minor Ball)",
    5: "滚珠中度故障_0.014 (Moderate Ball)",
    6: "滚珠重度故障_0.021 (Severe Ball)",
    7: "外圈轻度故障_0.007 (Minor Outer Race)",
    8: "外圈中度故障_0.014 (Moderate Outer Race)",
    9: "外圈重度故障_0.021 (Severe Outer Race)",
}

def extract_label(text: str) -> Optional[str]:
    """
    从模型生成的文本中提取 CWRU 10 分类标签。
    策略 1：完整包含中英文名称
    策略 2：退阶匹配仅包含中文部分
    """
    # 策略 1：匹配全称
    for cls_id, cls_str in CLASS_MAP_ZH.items():
        if cls_str in text:
            return cls_str
            
    # 策略 2：退阶匹配 (仅匹配中文部分，如 "内圈轻度故障_0.007")
    for cls_id, cls_str in CLASS_MAP_ZH.items():
        zh_part = cls_str.split(" (")[0]
        if zh_part in text:
            return cls_str
            
    return None

def true_label_str(label_id: int) -> str:
    """将 true_label int 映射为标准字符串。"""
    return CLASS_MAP_ZH[label_id]

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

        # AlignmentLayer: input_dim 必须等于你的 Deep Feature 维度 (当前架构默认 64)
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

        output_ids = self.model.generate(
            inputs_embeds    = inputs_embeds,
            attention_mask   = extended_mask,
            max_new_tokens   = max_new_tokens,
            do_sample        = False,
            pad_token_id     = self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

def evaluate(checkpoint_dir: str, json_path: str, pt_path: str,
             qwen_path: str = QWEN_PATH, max_new_tokens: int = 256):
    
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    features = torch.load(pt_path, map_location='cpu')   
    assert len(records) == features.shape[0]

    evaluator = MultimodalEvaluator(checkpoint_dir, qwen_path)

    total   = len(records)
    correct = 0
    # 更新为 10 分类的统计字典
    per_class_correct = {i: 0 for i in range(10)}
    per_class_total   = {i: 0 for i in range(10)}
    failures = []   

    print(f"\n[EVAL] 开始评估 {total} 条样本...")
    for i in tqdm(range(total)):
        rec  = records[i]
        feat = features[i]   

        true_id  = rec['true_label']
        true_str = true_label_str(true_id)
        per_class_total[true_id] += 1

        generated = evaluator.generate_one(
            deep_feature = feat,
            instruction  = rec['instruction'],
            input_ctx    = rec['input'],
            max_new_tokens = max_new_tokens,
        )

        pred_str = extract_label(generated)

        if pred_str == true_str:
            correct += 1
            per_class_correct[true_id] += 1
        else:
            failures.append({
                "idx": i, "true": true_str, "pred": pred_str,
                "generated": generated[:200]
            })

    accuracy = correct / total * 100
    print(f"\n{'='*70}")
    print(f"  CWRU 多模态 LLM 轴承诊断评估结果 (10 分类)")
    print(f"{'='*70}")
    print(f"  总样本数        : {total}")
    print(f"  正确预测        : {correct}")
    print(f"  诊断准确率      : {accuracy:.2f}%")
    print(f"{'='*70}")
    print("  Per-class 准确率：")
    for cls_id, cls_name in CLASS_MAP_ZH.items():
        n  = per_class_total[cls_id]
        c  = per_class_correct[cls_id]
        pct = c / n * 100 if n > 0 else 0.0
        # 调整了对其格式，容纳更长的类名
        print(f"    [{cls_id}] {cls_name:<40} {c:>4}/{n:<4}  ({pct:.1f}%)")
    print(f"{'='*70}")

    if failures:
        print(f"\n  ⚠️  解析失败 / 错误预测：{len(failures)} 条（前 5 条）")
        for f in failures[:5]:
            print(f"    idx={f['idx']} | true={f['true']} | pred={f['pred']}")
            print(f"    generated: {f['generated']}...")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态轴承诊断 LLM 评估脚本 (CWRU)")
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
        checkpoint_dir = args.checkpoint,
        json_path      = args.test_json,
        pt_path        = args.test_pt,
        qwen_path      = args.qwen_path,
        max_new_tokens = args.max_new_tokens,
    )