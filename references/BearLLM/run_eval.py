import os
import json
import torch
import numpy as np
from tqdm import tqdm
from dotenv import dotenv_values
from peft import PeftModel
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

from src.fine_tuning import description_len, signal_token_id, get_bearllm, mod_xt_for_qwen

# 加载环境变量
env = dotenv_values('.env')
qwen_weights = env['QWEN_WEIGHTS']
bearllm_weights = env['BEARLLM_WEIGHTS']
processed_dir = env.get('PROCESSED_DIR', './data/processed')
active_dataset = env.get('ACTIVE_DATASET', 'mbhm')

# 动态加载对应数据集的标签库及缓存名称
if active_dataset == 'cwru':
    from functions.cwru import DESCRIPTION_TEXT
    cache_name = 'cwru_dataset.json'
    signals_name = 'cwru_signals.npy'
elif active_dataset == 'dirg':
    from functions.dirg import DESCRIPTION_TEXT
    cache_name = 'dirg_dataset.json'
    signals_name = 'dirg_signals.npy'
elif active_dataset == 'xjtu':
    from functions.xjtu import DESCRIPTION_TEXT
    cache_name = 'xjtu_dataset.json'
    signals_name = 'xjtu_signals.npy'
else:
    from functions.mbhm import DESCRIPTION_TEXT
    cache_name = 'dataset.json'
    signals_name = 'data.hdf5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_eval():
    cache_path = os.path.join(processed_dir, cache_name)
    signals_path = os.path.join(processed_dir, signals_name)

    if not os.path.exists(cache_path) or not os.path.exists(signals_path):
        print(f"[ERROR] 找不到 {active_dataset.upper()} 的数据缓存！请先运行预训练脚本生成切分数据。")
        return

    # 1. 加载严格对齐的测试集数据 (与对比方法一模一样的 X_test.npy 映射)
    with open(cache_path, 'r') as f:
        dataset_info = json.load(f)
    test_samples = dataset_info['test']
    
    # 使用 mmap 节约内存
    signals = np.load(signals_path, mmap_mode='r')

    print(f"[*] 当前使用数据集: {active_dataset.upper()}")
    print(f"[*] 发现测试集样本数量: {len(test_samples)} (该数量与 X_test.npy 完全一致)")
    print("[*] 正在加载基座大模型与 LoRA 权重，请稍候...")

    # 2. 初始化分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = get_bearllm(train_mode=False)
    model = PeftModel.from_pretrained(model, f'{bearllm_weights}/lora')
    model.eval()

    # 3. 定义评估指标容器
    correct_count = 0
    hallucination_count = 0
    y_true = []
    y_pred = []
    
    # 标准回答前缀，用于评估忠实度
    expected_prefix = "Based on the unified vibration signal representation, the bearing exhibits a"
    instruction_template = "The dynamic sensor captured this vibration signal. Can you analyze the bearing status based on it? #state_place_holder#"

    # 【核心防抖】为了防止子串匹配漏洞，将标签按长度降序排列
    sorted_states = sorted(DESCRIPTION_TEXT, key=len, reverse=True)

    print("[*] 开始进行端到端严格推理与评估...")
    
    # 加入 mininterval 和 ascii 保证输出到 log 文本时不会因为频繁刷新而乱码
    for idx, (file_id, ref_id, label) in enumerate(tqdm(test_samples, mininterval=10, ascii=True, desc="评估进度")):
        # -- 数据准备 --
        query_data = signals[file_id]
        ref_data = signals[ref_id]
        rv = np.array([query_data, ref_data])
        np.save('./cache.npy', rv)

        # 获取当前样本的真实标签文本 (Ground Truth)
        gt_state = DESCRIPTION_TEXT[label]

        # -- Prompt 构造 --
        place_holder_ids = torch.ones(description_len, dtype=torch.long) * signal_token_id
        text_part1, text_part2 = mod_xt_for_qwen(instruction_template)

        user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids]).to(device)
        attention_mask = torch.ones_like(user_ids).to(device)

        # -- 模型推理 --
        with torch.no_grad():
            output = model.generate(
                user_ids.unsqueeze(0), 
                attention_mask=attention_mask.unsqueeze(0), 
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True).strip()

        # -- 严格指标计算逻辑 --
        predicted_state = "Unknown (Hallucination)"
        
        for state in sorted_states:
            if state in output_text:
                predicted_state = state
                break  # 匹配到最长的就跳出，防止子串重叠漏洞
                
        if predicted_state == gt_state:
            correct_count += 1
            
        y_true.append(gt_state)
        y_pred.append(predicted_state)

        # 幻觉判断
        if (expected_prefix not in output_text) or (predicted_state == "Unknown (Hallucination)") or (len(output_text) > 150):
            hallucination_count += 1

    # -- 输出最终报告 --
    total = len(test_samples)
    accuracy = (correct_count / total) * 100
    hallucination_rate = (hallucination_count / total) * 100

    print("\n" + "="*60)
    print(f" 📊 BearLLM ({active_dataset.upper()}) 公平严格评估报告 📊")
    print("="*60)
    print(f" 测试样本总数    : {total}")
    print(f" 真实准确率 (Acc): {accuracy:.2f}% ({correct_count}/{total})")
    print(f" 幻觉率 (Halluc.): {hallucination_rate:.2f}% ({hallucination_count}/{total})")
    print("-" * 60)
    print("【分类性能详细报告 (Precision / Recall / F1)】")
    # 过滤掉不存在的保留类别
    valid_labels = [s for s in DESCRIPTION_TEXT if s in y_true or s in y_pred]
    print(classification_report(y_true, y_pred, labels=valid_labels, zero_division=0))
    print("="*60)
    
    # 清理缓存
    if os.path.exists('./cache.npy'):
        os.remove('./cache.npy')

if __name__ == "__main__":
    run_eval()