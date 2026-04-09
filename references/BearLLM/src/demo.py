import torch
import os
import random
from dotenv import dotenv_values
from peft import PeftModel
from transformers import AutoTokenizer
from src.fine_tuning import description_len, signal_token_id, get_bearllm, mod_xt_for_qwen
import numpy as np
from functions.dcn import dcn
import json

env = dotenv_values()
mbhm_dataset = env['MBHM_DATASET']
qwen_weights = env['QWEN_WEIGHTS']
bearllm_weights = env['BEARLLM_WEIGHTS']
active_dataset = env.get('ACTIVE_DATASET', 'mbhm')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_cache():
    if active_dataset == 'cwru':
        # 从 CWRU 生成的测试集中随机抽取
        corpus_path = os.path.join(env.get('CWRU_PROCESSED', './data/processed'), 'cwru_corpus.json')
        signals_path = os.path.join(env.get('CWRU_PROCESSED', './data/processed'), 'cwru_signals.npy')
        
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
        signals = np.load(signals_path, mmap_mode='r')
        
        sample = random.choice(corpus)
        query_data = signals[sample['vib_id']]
        ref_data = signals[sample['ref_id']]
        demo_instruction = sample['instruction']
    else:
        # 原有 MBHM 逻辑
        demo_data = json.load(open(f'{mbhm_dataset}/demo_data.json'))
        query_data = dcn(np.array(demo_data['vib_data']))
        ref_data = dcn(np.array(demo_data['ref_data']))
        demo_instruction = demo_data['instruction']

    rv = np.array([query_data, ref_data])
    np.save('./cache.npy', rv)
    return demo_instruction

def run_demo():
    demo_instruction = create_cache()
    
    place_holder_ids = torch.ones(description_len, dtype=torch.long) * signal_token_id
    text_part1, text_part2 = mod_xt_for_qwen(demo_instruction)

    tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    user_part1_ids = tokenizer(text_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_part2_ids = tokenizer(text_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
    user_ids = torch.cat([user_part1_ids, place_holder_ids, user_part2_ids])
    user_ids = user_ids.to(device)
    attention_mask = torch.ones_like(user_ids)
    attention_mask = attention_mask.to(device)

    model = get_bearllm(train_mode=False)
    model = PeftModel.from_pretrained(model, f'{bearllm_weights}/lora') # 注意原版demo少了/lora，这里修正
    model.eval()

    output = model.generate(user_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), max_new_tokens=2048)
    output_text = tokenizer.decode(output[0, user_ids.shape[0]:], skip_special_tokens=True)
    
    print("\n--- 指令输入 ---")
    print(demo_instruction.replace('#state_place_holder#', '[VIBRATION_SIGNAL]'))
    print("\n--- LLM 诊断输出 ---")
    print(output_text)

if __name__ == "__main__":
    run_demo()