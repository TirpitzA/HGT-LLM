"""
generate_multimodal_dataset.py
=============================================
基于预训练的物理骨干网，按照 7:2:1 物理隔离生成多模态 SFT 数据集。
通过 --config 动态加载任意数据集参数。
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def split_dig_explanation(explanation, prediction, critical_slice_idx, time_score, target_class_zh, forbidden_words):
    """ 将 DIG 报告拆分为输入上下文和输出推理链 """
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

    # 安全检查：根据配置的 forbidden_words 进行过滤
    for fw in forbidden_words:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 运行设备: {DEVICE}")

    # 1. 动态加载模型与工具类
    from utils.model_loader import load_pretrained_physics_net
    from utils.dig_construction import BearingDynamicInstanceGraph
    
    print(f"[INFO] 正在加载配置: {args.config}")
    try:
        model, edge_index, config = load_pretrained_physics_net(args.config, DEVICE)
    except FileNotFoundError as e:
        print(e)
        return

    dataset_name = config.get("dataset_name", "Unknown")
    class_map = config.get("labels", {})
    forbidden_words = config.get("forbidden_words", [])

    A = edge_index # Here edge_index is returned, but we need A logic if requested directly in DIG construction. Wait, BearingDynamicInstanceGraph takes physics_adjacency! But we return edge_index from loader. Let's fix this in model loader or re-import it here.
    from utils.physics_graph import get_bearing_physics_adjacency
    A, _ = get_bearing_physics_adjacency()

    dig_builder = BearingDynamicInstanceGraph(physics_adjacency=A, config=config)

    print(f"[INFO] {dataset_name} ({len(class_map)}分类) 物理骨干网已加载，准备生成数据...")

    instruction_template = ("【background：领域知识库注入】作为专家级智能体，请结合输入的底层振动语义特征向量，"
                            "以及以下时空图注意力解析，输出诊断结论与推理链路：")

    # 2. 循环处理 7:2:1 物理隔离的分支
    for split in ['train', 'val', 'test']:
        print(f"\n[PROCESS] 正在为 {dataset_name} {split.upper()} 集生成多模态对齐数据...")
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

                deep_feat = output['deep_feature'].cpu() 
                pred_label = int(output['logits'].argmax(dim=-1).item())

                # 仅在训练集和验证集跳过错误样本，测试集必须保留全量真实数据
                if split in ['train', 'val'] and pred_label != y:
                    skipped += 1
                    continue

                confidence = float(torch.softmax(output['logits'], dim=-1)[0, pred_label].item() * 100)
                dig_result = dig_builder.process_sample(
                    prediction=pred_label, confidence=confidence,
                    edge_attention=output['edge_weights'].cpu().numpy()[0],
                    slice_attention=output['slice_weights'].cpu().numpy()[0], lang='zh'
                )

                input_ctx, output_txt = split_dig_explanation(
                    dig_result['explanation'], pred_label, dig_result['critical_time_slice'],
                    float(output['slice_weights'].cpu().numpy()[0][dig_result['critical_time_slice']]), 
                    class_map[y], forbidden_words
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
            print(f"[INFO] {split} 完成: 有效样本 {len(qa_pairs)}, 跳过错误样本 {skipped}")

    print(f"\n✅ {dataset_name} 多模态数据集生成完毕！请直接运行 python src/train.py 进行微调。")

if __name__ == "__main__":
    main()