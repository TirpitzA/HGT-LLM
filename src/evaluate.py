"""
evaluate.py (Token Replacement 版)
==================================
多模态模型评估脚本。
改动：适配 token 替换逻辑。在 input_ids 中插入 SIGNAL_TOKEN_ID。
"""
import os
import sys
import json
import yaml
import argparse
import math
import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
try:
    from thop import profile
except ImportError:
    profile = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multimodal_qwen import AlignmentLayer, ModifiedEmbedding, SIGNAL_TOKEN_ID, DEFAULT_NUM_VIB_TOKENS
from src.zero_shot_adapter import ZeroShotChannelAdapter

QWEN_PATH = os.path.join(PROJECT_ROOT, "qwen_weights")
DATA_DIR  = PROJECT_ROOT

class MultimodalEvaluator:
    SYSTEM_PROMPT = (
        "<|im_start|>system\n"
        "你是一名专业的轴承故障诊断专家。结合输入的底层振动信号语义特征和时空图谱分析，"
        "请给出严谨的推理过程与最终诊断结论。\n<|im_end|>\n"
    )

    def __init__(self, checkpoint_dir: str, qwen_path: str = QWEN_PATH, 
                 num_vib_tokens: int = DEFAULT_NUM_VIB_TOKENS,
                 signal_token_id: int = SIGNAL_TOKEN_ID):
        self.num_vib_tokens = num_vib_tokens
        self.signal_token_id = signal_token_id
        
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
        
        # 替换 Embedding 为 Token 替换模式
        original_embedding = self.model.get_input_embeddings()
        self.modified_embedding = ModifiedEmbedding(
            original_embedding, self.alignment_layer,
            signal_token_id, num_vib_tokens
        )
        self.model.set_input_embeddings(self.modified_embedding)
        
        self._print_model_complexity()
        print("[EVAL] 模型就绪✓ (Token Replacement Mode)")

    def _print_model_complexity(self):
        trainable_params, all_param = self.model.get_nb_trainable_parameters()
        align_params = sum(p.numel() for p in self.alignment_layer.parameters())
        
        total_trainable = trainable_params + align_params
        print(f"\n[COMPLEXITY] 📊 算力与参数量分析:")
        print(f" - 冻结基座参数量 (Base LLM) : {all_param - trainable_params:,} (约 {all_param/1e9:.1f} B)")
        print(f" - LoRA 可训练参数量        : {trainable_params:,}")
        print(f" - 视觉对齐层 (Alignment)   : {align_params:,}")
        print(f" - 总计可训练参数量 (Ours)  : {total_trainable:,} (占比 {(total_trainable/all_param)*100:.3f}%)")
        
        if profile is not None:
            # 评估对齐层 (Alignment Layer) 的运算量
            dummy_feat = torch.randn(1, 64).to(self.device).to(torch.float16)
            with torch.no_grad():
                align_flops, _ = profile(self.alignment_layer, inputs=(dummy_feat,), verbose=False)
            
            # LLM 部分通过理论公式估算: GFLOPS ≈ 2 * Params * Tokens
            # 取典型序列长度 L=128
            seq_len = 128
            llm_params = all_param # 总参数量
            # 理论估算 (1 token forward): 2 * params
            # 为了严谨, 我们仅展示对齐层实测 + LLM 规模参数
            gflops_align = align_flops / 1e9
            print(f" - [Alignment] 运算量          : {gflops_align:.6f} GFLOPS")
            print(f" - [Base LLM]  推理估算 (L=1次) : 约 {2 * all_param / 1e9:.2f} GFLOPS (理论值)")
            print(f" - [Total]     系统级算力需求   : 约 {gflops_align + 2 * all_param / 1e9:.2f} GFLOPS\n")
        else:
            print(" - [!] 未安装 thop，跳过 GFLOPS 详细测试。\n")

    @torch.no_grad()
    def compute_ppl(self, deep_feature: torch.Tensor, instruction: str,
                    input_ctx: str, output_txt: str) -> float:
        self.model.eval()
        self.alignment_layer.eval()
        
        if deep_feature.dim() == 1:
            deep_feature = deep_feature.unsqueeze(0)
        deep_feature = deep_feature.to(self.device).to(torch.float16)
        
        # 将特征写入缓存
        self.modified_embedding.set_feature(deep_feature)

        user_text = (
            f"<|im_start|>user\n"
            f"{instruction}\n\n"
            f"{input_ctx}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_prompt = self.SYSTEM_PROMPT + user_text

        prompt_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(output_txt + "<|im_end|>", add_special_tokens=False).input_ids
        
        # 在 prompt 开头插入占位符
        signal_placeholder = [self.signal_token_id] * self.num_vib_tokens
        prompt_ids = signal_placeholder + prompt_ids
        
        seq_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(seq_ids)
        
        vib_labels = [-100] * self.num_vib_tokens
        prompt_labels = [-100] * (len(prompt_ids) - self.num_vib_tokens)
        extended_labels = torch.tensor([vib_labels + prompt_labels + target_ids], dtype=torch.long, device=self.device)
        
        outputs = self.model(
            input_ids=seq_ids,
            attention_mask=attention_mask,
            labels=extended_labels
        )
        
        loss = outputs.loss
        ppl = torch.exp(loss)
        return ppl.item()

    @torch.no_grad()
    def generate_one(self, deep_feature: torch.Tensor, instruction: str,
                     input_ctx: str, max_new_tokens: int = 256) -> str:
        
        if deep_feature.dim() == 1:
            deep_feature = deep_feature.unsqueeze(0)
        deep_feature = deep_feature.to(self.device).to(torch.float16)

        # 将特征写入缓存
        self.modified_embedding.set_feature(deep_feature)

        user_text = (
            f"<|im_start|>user\n"
            f"{instruction}\n\n"
            f"{input_ctx}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_prompt = self.SYSTEM_PROMPT + user_text

        prompt_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids
        signal_placeholder = [self.signal_token_id] * self.num_vib_tokens
        prompt_ids = signal_placeholder + prompt_ids
        
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.model.generate(
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            max_new_tokens   = max_new_tokens,
            do_sample        = False,
            pad_token_id     = self.tokenizer.eos_token_id,
        )
        
        # 切片截断 prompt 部分
        output_ids = output_ids[0][input_ids.shape[1]:]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


def inject_awgn_noise(feature_tensor: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = torch.mean(feature_tensor ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(feature_tensor) * math.sqrt(noise_power)
    return feature_tensor + noise

def extract_label(text: str, class_map: dict) -> Optional[str]:
    for cls_id, cls_str in class_map.items():
        if cls_str in text:
            return cls_str
    for cls_id, cls_str in class_map.items():
        zh_part = cls_str.split(" (")[0]
        if zh_part in text:
            return cls_str
    return None

def evaluate(config_path: str, checkpoint_dir: str, json_path: str, pt_path: str,
             qwen_path: str = QWEN_PATH, max_new_tokens: int = 256, snr: Optional[float] = None):
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_name = config.get("dataset_name", "Unknown")
    class_map = config.get("labels", {})
    num_classes = config["model_params"]["num_classes"]

    # 0. 数据路径解析
    data_dir = config.get("data_dir", "data/")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    
    # 路径解析增强：优先使用 data_dir 下的文件，除非命令行显式指定了非默认路径
    default_json = os.path.join(PROJECT_ROOT, "data", "test_sft.json")
    default_pt = os.path.join(PROJECT_ROOT, "data", "test_features.pt")

    if json_path is None or json_path == default_json:
        alt_json = os.path.join(data_dir, "test_sft.json")
        if os.path.exists(alt_json):
            json_path = alt_json
            
    if pt_path is None or pt_path == default_pt:
        alt_pt = os.path.join(data_dir, "test_features.pt")
        if os.path.exists(alt_pt):
            pt_path = alt_pt

    # 如果此时路径仍不存在，抛出更有意义的错误
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到测试 JSON 文件: {json_path}")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"未找到测试特征文件: {pt_path}")

    print(f"[EVAL] 使用测试数据: {json_path}")

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
    y_true_ids = []
    y_pred_ids = []
    total_ppl = 0.0
    
    # 逆向映射用于计算 F1
    rev_class_map = {v: k for k, v in class_map.items()}

    snr_msg = f" (附加特征域 AWGN 噪声, SNR={snr}dB)" if snr is not None else " (干净测试集)"
    print(f"\n[EVAL] 开始评估 {total} 条 {dataset_name} 集样本{snr_msg}...")
    
    for i in tqdm(range(total)):
        rec  = records[i]
        feat = features[i].clone()   

        if snr is not None:
            feat = inject_awgn_noise(feat, snr)

        # 检查特征维度是否需要适配 (Zero-shot Robustness)
        # HGT 期望的输入形状通常包含通道维 [Slices, Length, Channels]
        # 但如果是 Alignment Layer 的输入 deep_feature [64], 则是在 feature extraction 之后
        # 这里 feat 是 deep_feature [64], 所以不需要 channel adapter
        # Channel adapter 主要用于 evaluate_physics 的 backbone 阶段
        
        true_id  = rec['true_label']
        true_str = class_map[true_id]
        per_class_total[true_id] += 1

        ppl_score = evaluator.compute_ppl(
            deep_feature=feat,
            instruction=rec['instruction'],
            input_ctx=rec['input'],
            output_txt=rec['output']
        )
        total_ppl += ppl_score

        generated = evaluator.generate_one(
            deep_feature = feat,
            instruction  = rec['instruction'],
            input_ctx    = rec['input'],
            max_new_tokens = max_new_tokens,
        )

        pred_str = extract_label(generated, class_map)
        y_true_ids.append(true_id)
        
        if pred_str is not None:
            pred_id = rev_class_map[pred_str]
            y_pred_ids.append(pred_id)
        else:
            # 解析失败，视为随机分类或者一个不存在的类
            y_pred_ids.append(-1) 

        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

        if pred_str == true_str:
            correct += 1
            per_class_correct[true_id] += 1
        else:
            failures.append({
                "idx": i, "true": true_str, "pred": pred_str,
                "generated": generated[:200]
            })

    from sklearn.metrics import precision_recall_fscore_support
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true_ids, y_pred_ids, average='macro', zero_division=0)

    accuracy = correct / total * 100
    avg_ppl = total_ppl / total

    # 保存结果供 benchmark_master.py 读取
    hgt_llm_results = {
        "HGT-LLM": {
            "Accuracy": accuracy / 100.0,
            "F1": f1_macro,
            "PPL": avg_ppl,
            "Details": {class_map[i]: (per_class_correct[i]/per_class_total[i] if per_class_total[i]>0 else 0) for i in class_map}
        }
    }
    result_file = os.path.join(PROJECT_ROOT, "results", f"benchmark_{dataset_name}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(hgt_llm_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  {dataset_name} 多模态 LLM 轴承诊断评估结果 ({num_classes} 分类)")
    print(f"{'='*70}")
    print(f"  工况环境        : {snr_msg}")
    print(f"  总样本数        : {total}")
    print(f"  正确预测        : {correct}")
    print(f"  诊断准确率      : {accuracy:.2f}%")
    print(f"  F1 Score(Macro) : {f1_macro:.4f}")
    print(f"  平均 PPL        : {avg_ppl:.4f}")
    print(f"  结果已保存至    : {result_file}")
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
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(DATA_DIR, "bearllm_weights/checkpoints_cwru/best_checkpoint"),
                        help="Checkpoint 目录（内含 lora_adapter/ + alignment_layer.pt）")
    parser.add_argument("--test_json", type=str,
                        default=os.path.join(DATA_DIR, "data", "test_sft.json"),
                        help="测试数据 JSON 路径")
    parser.add_argument("--test_pt",   type=str,
                        default=os.path.join(DATA_DIR, "data", "test_features.pt"),
                        help="测试 deep_feature .pt 路径")
    parser.add_argument("--qwen_path", type=str, default=QWEN_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--snr", type=float, default=None, 
                        help="注入的 AWGN 噪声强度(dB)。例如: 5, 0, -5。不传则为纯净测试。")
    
    args = parser.parse_args()

    evaluate(
        config_path    = args.config,
        checkpoint_dir = args.checkpoint,
        json_path      = args.test_json,
        pt_path        = args.test_pt,
        qwen_path      = args.qwen_path,
        max_new_tokens = args.max_new_tokens,
        snr            = args.snr,
    )