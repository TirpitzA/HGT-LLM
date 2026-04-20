"""
evaluate_bearllm.py
===================
为 BearLLM 深度定制的公平统一评估脚本。
复用 HGT-LLM 的 X_test.npy 纯净数据管线，但底层调用 BearLLM 的 FCN 和 ModifiedEmbedding。
包含 GFLOPs、PPL、SNR 鲁棒性注入。
"""
import os
import sys
import json
import yaml
import argparse
import math
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from thop import profile, clearing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入 BearLLM 私有组件 (通过 references/BearLLM)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references/BearLLM"))
from references.BearLLM.src.fine_tuning import get_bearllm, mod_xt_for_qwen, SIGNAL_TOKEN_ID, DESCRIPTION_LEN

QWEN_PATH = os.path.join(PROJECT_ROOT, "qwen_weights")
BEARLLM_WEIGHTS = os.path.join(PROJECT_ROOT, "references/BearLLM/bearllm_weights")
DATA_DIR = PROJECT_ROOT

def inject_awgn_noise(signal_tensor: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = torch.mean(signal_tensor ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(signal_tensor) * math.sqrt(noise_power)
    return signal_tensor + noise

def extract_label(text: str, class_map: dict) -> Optional[str]:
    # BearLLM 的 GT 描述文本不一定严格和 YAML labels 一致
    # 尝试各种匹配策略
    for cls_id, cls_str in class_map.items():
        if cls_str in text:
            return cls_str
        zh_part = cls_str.split(" (")[0]
        if zh_part in text:
            return cls_str
    
    # 退阶匹配：BearLLM 对于 CWRU 有特定的英文描述
    legacy_map = [
        "Fault-Free", 
        "Minor Inner Ring Fault", "Moderate Inner Ring Fault", "Severe Inner Ring Fault",
        "Minor Ball Fault", "Moderate Ball Fault", "Severe Ball Fault",
        "Minor Outer Ring Fault", "Moderate Outer Ring Fault", "Severe Outer Ring Fault"
    ]
    for s in legacy_map:
        if s in text:
            return s
            
    return None

class BearLLMEvaluator:
    def __init__(self, qwen_path=QWEN_PATH, bearllm_weights=BEARLLM_WEIGHTS):
        # 强制设置当前工作目录以让 dotenv 能找到 BearLLM/.env
        os.environ['QWEN_WEIGHTS'] = qwen_path
        os.environ['BEARLLM_WEIGHTS'] = bearllm_weights
        os.environ['DESCRIPTION_LEN'] = "5"
        os.environ['LLM_HIDDEN_SIZE'] = "1536"
        os.environ['SIGNAL_TOKEN_ID'] = "151925"
        
        print("[EVAL] 加载 BearLLM (Qwen2-1.5B + LoRA + AlignmentAdapter)...")
        # get_bearllm() 内部会实例化 FCN, AlignmentLayer 和 ModifiedEmbedding
        self.model = get_bearllm(train_mode=False) 
        self.model = PeftModel.from_pretrained(self.model, os.path.join(bearllm_weights, "lora"))
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.device = next(self.model.parameters()).device
        self._print_complexity()
        
    def _print_complexity(self):
        trainable_params, all_param = self.model.get_nb_trainable_parameters()
        print(f"\n[COMPLEXITY] 📊 BearLLM 算力分析:")
        print(f" - Base LLM: {all_param - trainable_params:,}")
        print(f" - LoRA Params: {trainable_params:,}")
        
        # 获取 AlignmentAdapter
        adapter = self.model.get_input_embeddings().adapter
        adapter_params = sum(p.numel() for p in adapter.parameters())
        print(f" - AlignmentAdapter (FCN + Align): {adapter_params:,}")
        
        dummy_input = torch.randn(1, 2, 24000).to(self.device).to(torch.float32)
        macs, params = profile(adapter, inputs=(dummy_input,), verbose=False)
        clearing(adapter)
        print(f" - 视觉编码器计算量 (GFLOPs): {macs / 1e9 * 2:.3f} GFLOPs\n")

    @torch.no_grad()
    def compute_ppl(self, instruction: str, target_text: str, dummy_signal_ids: torch.Tensor) -> float:
        """
        PPL 计算要求模型不能只是 generate，而是要在 forward 时给 labels。
        对于 BearLLM 的 `mod_xt_for_qwen` 来说，它会自动拼接特殊 token。
        """
        self.model.eval()
        
        text_part1, text_part2 = mod_xt_for_qwen(instruction)
        part1_ids = self.tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        part2_ids = self.tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        
        user_ids = torch.cat([part1_ids, dummy_signal_ids, part2_ids]).to(self.device)
        target_ids = self.tokenizer(target_text + "<|im_end|>", return_tensors='pt', add_special_tokens=False).input_ids[0].to(self.device)
        
        input_ids = torch.cat([user_ids, target_ids]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        
        # 构造 labels：用户输入部分（含占位符）为 -100，只对 target 计算 loss
        labels = torch.cat([
            torch.full_like(user_ids, -100),
            target_ids
        ]).unsqueeze(0)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return torch.exp(outputs.loss).item()

    @torch.no_grad()
    def generate_one(self, instruction: str, dummy_signal_ids: torch.Tensor, max_new_tokens=50) -> str:
        self.model.eval()
        
        text_part1, text_part2 = mod_xt_for_qwen(instruction)
        part1_ids = self.tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        part2_ids = self.tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        
        user_ids = torch.cat([part1_ids, dummy_signal_ids, part2_ids]).to(self.device)
        attention_mask = torch.ones_like(user_ids)
        
        outputs = self.model.generate(
            user_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0, user_ids.shape[0]:], skip_special_tokens=True).strip()


def evaluate_bearllm(config_path: str, qwen_path: str, bearllm_weights: str, snr: Optional[float] = None):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    dataset_name = config.get("dataset_name", "Unknown")
    class_map = config.get("labels", {})
    
    # BearLLM 需要原始的时序振动信号（且长度需要重采样为 24000）
    print("[EVAL] 加载测试集原始信号 X_test.npy / y_test.npy ...")
    x_path = os.path.join(DATA_DIR, "data", "X_test.npy")
    y_path = os.path.join(DATA_DIR, "data", "y_test.npy")
    
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    
    # 找到一个健康样本作为 reference (Label=0)
    health_idx = np.where(y_test == 0)[0]
    if len(health_idx) > 0:
        ref_signal = X_test[health_idx[0]].flatten()
    else:
        print("[WARNING] 测试集中找不到健康状态(label=0)作为 Reference！使用第一个样本替代。")
        ref_signal = X_test[0].flatten()

    def process_signal_for_bearllm(raw_signal, length=24000):
        # 展平 + Tile 保证长度
        sig = raw_signal.flatten()
        if len(sig) < length:
            repeats = (length // len(sig)) + 1
            sig = np.tile(sig, repeats)[:length]
        else:
            sig = sig[:length]
        
        # dcn() 归一化 (此处由于我们在脚本内，简化实现)
        sig = sig - np.mean(sig)
        max_val = np.max(np.abs(sig))
        if max_val > 0:
            sig = sig / max_val
        return sig

    print(f"[EVAL] 正在对 {len(X_test)} 个样本进行(2, 24000)重采样合并...")
    ref_processed = process_signal_for_bearllm(ref_signal)
    
    evaluator = BearLLMEvaluator(qwen_path, bearllm_weights)
    
    total = len(X_test)
    correct = 0
    total_ppl = 0.0
    
    place_holder_ids = torch.ones(DESCRIPTION_LEN, dtype=torch.long) * SIGNAL_TOKEN_ID
    instruction_template = "The dynamic sensor captured this vibration signal. Can you analyze the bearing status based on it? #state_place_holder#"

    snr_msg = f"(附加 AWGN 噪声, SNR={snr}dB)" if snr is not None else "(干净测试集)"
    print(f"\n[EVAL] 🚀 开始 BearLLM 端到端严格推理 {snr_msg}")
    
    for i in tqdm(range(total)):
        query_raw = X_test[i]
        
        if snr is not None:
            # 注：噪声应当注入在原始尺度，然后再做 BearLLM特供的归一化
            query_raw = inject_awgn_noise(torch.tensor(query_raw), snr).numpy()
            
        query_processed = process_signal_for_bearllm(query_raw)
        
        # 组成 (2, 24000) 阵列并缓存到 cache.npy 供 ModifiedEmbedding 读取
        combined = np.array([query_processed, ref_processed])
        np.save('./cache.npy', combined)
        
        pred_id = int(y_test[i])
        true_desc = extract_label(class_map.get(pred_id, ""), class_map) or "Unknown"
        # 为 PPL 提供一个目标文本示例 (这部分仅做 PPL 评估，不影响生成)
        target_response = f"Based on the unified vibration signal representation, the bearing exhibits a {true_desc}."

        # 计算 PPL
        ppl = evaluator.compute_ppl(
            instruction=instruction_template, 
            target_text=target_response, 
            dummy_signal_ids=place_holder_ids
        )
        total_ppl += ppl

        # 自由生成
        generated = evaluator.generate_one(
            instruction=instruction_template,
            dummy_signal_ids=place_holder_ids,
            max_new_tokens=50
        )
        
        # 抽取并比较
        pred_desc = extract_label(generated, class_map)
        if pred_desc and pred_desc in true_desc:
            correct += 1

    accuracy = correct / total * 100
    avg_ppl = total_ppl / total
    
    print(f"\n{'='*60}")
    print(f" 📊 BearLLM ({dataset_name}) 公平严格评估报告")
    print(f"{'='*60}")
    print(f" 工况        : {snr_msg}")
    print(f" 测试样本量  : {total}")
    print(f" 准确率 (Acc): {accuracy:.2f}% ({correct}/{total})")
    print(f" 平均似然(PPL): {avg_ppl:.4f}")
    print(f"{'='*60}\n")
    
    if os.path.exists('./cache.npy'):
        os.remove('./cache.npy')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--bearllm_weights", type=str, default=BEARLLM_WEIGHTS)
    parser.add_argument("--qwen", type=str, default=QWEN_PATH)
    parser.add_argument("--snr", type=float, default=None)
    args = parser.parse_args()
    
    evaluate_bearllm(args.config, args.qwen, args.bearllm_weights, args.snr)
