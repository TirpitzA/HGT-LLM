"""
generate_multimodal_dataset.py (CWRU Edition)
=============================================
基于预训练的 CWRU 物理骨干网，按照 7:2:1 物理隔离生成多模态 SFT 数据集。
适配 CWRU 10 分类任务：1 种健康 + 9 种故障（3 部件 x 3 尺寸）。
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'bearllm_weights', 'best_model.pth')
DATA_DIR        = os.path.join(PROJECT_ROOT, 'data')

# ── 标签映射 (适配 CWRU 10 分类) [cite: 432] ──────────────────────────────────
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

def split_dig_explanation(explanation, prediction, critical_slice_idx, time_score, target_class_zh):
    """ 将 DIG 报告拆分为输入上下文和输出推理链 (遵循 BearLLM 提示词设计) [cite: 57, 117] """
    lines = explanation.split('\n')
    # 去掉报告标题、置信度行和分隔线 (前 3 行)
    body_lines = lines[3:] 
    # 以 "3. 动态知识图谱传播逻辑" 为界分割输入与输出
    split_idx = next((j for j, line in enumerate(body_lines) if line.startswith('3.')), None)

    if split_idx is not None:
        input_lines = body_lines[:split_idx]
        output_lines = body_lines[split_idx:]
    else:
        input_lines = body_lines
        output_lines = []

    input_context = '\n'.join(input_lines).strip()
    # 安全检查：确保 input 中不含具体的故障诊断结论词，以防大模型通过文本直接“偷看”答案
    FORBIDDEN = ['预测状态', '置信度', '诊断结论', '内圈故障', '滚珠故障', '外圈故障', '健康']
    for fw in FORBIDDEN:
        if fw in input_context:
            input_context = '\n'.join(l for l in input_context.split('\n') if fw not in l)

    reasoning_chain = '\n'.join(output_lines).strip()
    if reasoning_chain:
        output_text = f"【专家推理过程】\n{reasoning_chain}\n\n【诊断结论】：{target_class_zh}"
    else:
        # 健康类：仅基于平稳的观察特征给出结论
        output_text = (f"【专家推理过程】\n各部件振动信噪比处于正常基线水平：模型主要关注时间切片 [{critical_slice_idx}]"
                       f"（重要性得分: {time_score:.2f}），该切片振动模式平稳。\n\n【诊断结论】：{target_class_zh}")
    return input_context, output_text

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 运行设备: {DEVICE}")

    # 1. 加载适配 CWRU 的预训练物理模型与 DIG 工具 [cite: 10, 205]
    from utils.cwru_physics_graph import get_bearing_physics_adjacency
    from models.cwru_physics_pipeline import load_pretrained_physics_net
    from utils.cwru_dig_construction import BearingDynamicInstanceGraph

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] 找不到模型权重: {CHECKPOINT_PATH}")
        return

    # 加载已训练好的 10 分类、1 通道 CWRU 物理骨干网
    model, edge_index = load_pretrained_physics_net(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    model.eval()
    print(f"[INFO] CWRU 物理骨干网已加载，准备提取 64 维深度特征...")

    A, _, component_names = get_bearing_physics_adjacency()
    dig_builder = BearingDynamicInstanceGraph(physics_adjacency=A, component_names=component_names)

    instruction_template = ("【background：领域知识库注入】作为专家级智能体，请结合输入的底层振动语义特征向量，"
                            "以及以下时空图注意力解析，输出诊断结论与推理链路：")

    # 2. 循环处理 7:2:1 物理隔离的分支 (Train, Val, Test)
    for split in ['train', 'val', 'test']:
        print(f"\n[PROCESS] 正在为 CWRU {split.upper()} 集生成多模态对齐数据...")
        x_path = os.path.join(DATA_DIR, f'X_{split}.npy')
        y_path = os.path.join(DATA_DIR, f'y_{split}.npy')
        
        if not os.path.exists(x_path):
            print(f"[SKIP] 未找到 {x_path}")
            continue

        X_np = np.load(x_path, mmap_mode='r')
        y_np = np.load(y_path, mmap_mode='r')
        
        qa_pairs = []
        deep_features = []
        skipped = 0

        with torch.no_grad():
            for i in tqdm(range(len(X_np)), desc=f"处理 {split}"):
                x, y = X_np[i], int(y_np[i])
                X_ten = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                output = model(X_ten, return_attention=True)

                # 提取物理特征向量并对齐到词空间 [cite: 10, 234]
                deep_feat = output['deep_feature'].cpu() 
                pred_label = int(output['logits'].argmax(dim=-1).item())

                # 关键过滤：仅保留预测正确的样本以确保 SFT 训练数据的逻辑正确性
       #         if pred_label != y:
       #             skipped += 1
       #             continue
                # 仅在训练集跳过错误样本，测试集必须保留全量真实数据
                if split in ['train', 'val'] and pred_label != y:
                    skipped += 1
                    continue

                # 通过 DIG 生成物理可解释的诊断报告 [cite: 117]
                confidence = float(torch.softmax(output['logits'], dim=-1)[0, pred_label].item() * 100)
                dig_result = dig_builder.process_sample(
                    prediction=pred_label, confidence=confidence,
                    edge_attention=output['edge_weights'].cpu().numpy()[0],
                    slice_attention=output['slice_weights'].cpu().numpy()[0], lang='zh'
                )

                # 分离输入与推理链
                input_ctx, output_txt = split_dig_explanation(
                    dig_result['explanation'], pred_label, dig_result['critical_time_slice'],
                    float(output['slice_weights'].cpu().numpy()[0][dig_result['critical_time_slice']]), 
                    CLASS_MAP_ZH[y]
                )

                qa_pairs.append({
                    "idx": i, "true_label": y, "pred_label": pred_label,
                    "instruction": instruction_template,
                    "input": f"=== 振动时空图谱注意力分析 ===\n{input_ctx}\n",
                    "output": output_txt
                })
                deep_features.append(deep_feat)

        # 3. 保存物理隔离的数据集
        with open(os.path.join(DATA_DIR, f"{split}_sft.json"), 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        
        if deep_features:
            torch.save(torch.cat(deep_features, dim=0), os.path.join(DATA_DIR, f"{split}_features.pt"))
            print(f"[INFO] {split} 完成: 有效样本 {len(qa_pairs)}, 跳过分类错误样本 {skipped}")

    print("\n✅ CWRU 多模态数据集生成完毕！请直接运行 python src/train.py 进行微调。")

if __name__ == "__main__":
    main()