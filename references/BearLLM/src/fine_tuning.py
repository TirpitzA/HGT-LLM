import os
import torch
from dotenv import dotenv_values
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
import numpy as np
from models.FCN import FeatureEncoder

env = dotenv_values()

# --- 动态选择数据集 ---
active_dataset = env.get('ACTIVE_DATASET', 'mbhm')
if active_dataset == 'cwru':
    from functions.cwru import CorpusDataset
    print(">>> 微调模式：已加载 CWRU 数据集管道")
else:
    from functions.mbhm import CorpusDataset
# --- 修改结束 ---

qwen_weights = env['QWEN_WEIGHTS']
bearllm_weights = env['BEARLLM_WEIGHTS']
fcn_weights = f'{bearllm_weights}/fcn'
l3_weights = f'{bearllm_weights}/l3.npy'
align_weights = f'{bearllm_weights}/align.pth'
adapter_weights = f'{bearllm_weights}/vibration_adapter.pth'

description_len = int(env['DESCRIPTION_LEN'])
llm_hidden_size = int(env['LLM_HIDDEN_SIZE'])
signal_token_id = int(env['SIGNAL_TOKEN_ID'])
m = [4**4, 4**3, 4**2, 4**1, 1]

class HyperParameters:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r = 4  # 低秩分解的秩
        self.lora_alpha = 32  # LoRA的alpha参数
        self.lora_dropout = 0.1  # Dropout比例
        
        # 【GPU 性能优化核心】
        self.per_device_train_batch_size = 8  # 增大单次喂入的样本量，压榨显存
        self.gradient_accumulation_steps = 2  # 降低梯度累积步数以平衡 Batch 扩大
        
        self.logging_steps = 10
        self.num_train_epochs = 50
        self.save_steps = 200
        self.learning_rate = 1e-4
        self.lr_scheduler_type = 'cosine'

description_text = [
    "Fault-Free",
    "Minor Inner Ring Fault",
    "Moderate Inner Ring Fault",
    "Severe Inner Ring Fault",
    "Minor Ball Fault",
    "Moderate Ball Fault",
    "Severe Ball Fault"
    "Minor Outer Ring Fault",
    "Moderate Outer Ring Fault",
    "Severe Outer Ring Fault"
]

"""
description tokens are encoded as follows:

description_text.append(" ")
for text in description_text:
    encoded_text = tokenizer.encode(text)
    print(encoded_text)
"""
description_tokens = [
    [220, 220, 220, 58780, 62890],
    [57024, 220, 36356, 21525, 59149],
    [68623, 349, 36356, 21525, 59149],
    [1514, 19289, 36356, 21525, 59149],
    [57024, 220, 12836, 220, 59149],
    [68623, 349, 12836, 220, 59149],
    [1514, 19289, 12836, 220, 59149],
    [57024, 220, 55197, 21525, 59149],
    [68623, 349, 55197, 21525, 59149],
    [1514, 19289, 55197, 21525, 59149]
]

sys_prompt = ("As an expert in bearing fault diagnosis with extensive knowledge in mechanical engineering and failure "
              "analysis, you can assess the state of bearings. Typically, bearing states are categorized as ["
              "Fault-Free, Minor Inner Ring Fault, Moderate Inner Ring Fault, Severe Inner Ring Fault, Minor Ball "
              "Fault, Moderate Ball Fault, Severe Ball Fault, Minor Outer Ring Fault, Moderate Outer Ring Fault, "
              "Severe Outer Ring Fault]. Based on the description of the bearing state, answer my questions.")


def initialize_l3_weight():
    llm = AutoModelForCausalLM.from_pretrained(qwen_weights)
    embedding = llm.get_input_embeddings()
    tokens = torch.tensor(description_tokens)
    embeds = embedding(tokens).to(torch.float32).detach().cpu().numpy()
    np.save(l3_weights, embeds)
    return embeds


def load_l3_weight():
    if not os.path.exists(l3_weights):
        return initialize_l3_weight()
    return np.load(l3_weights)


class AlignmentLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128 * 47, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        self.linear3 = nn.Linear(10, description_len*llm_hidden_size)

    def forward(self, x):
        x = x.view(-1, 47 * 128) # channel:47, len:128
        x = self.linear1(x) # [batch_size, 128]
        x = self.relu(x)
        x = self.linear2(x) # [batch_size, 10]
        x = self.softmax(x)
        x = self.linear3(x) # [batch_size, description_len*llm_hidden_size]
        x = x.reshape(x.size(0), description_len, llm_hidden_size) # [batch_size, description_len, llm_hidden_size]
        x = x.to(torch.bfloat16)
        return x

    def load_default(self):
        classifier_weights = torch.load(f'{fcn_weights}/classifier.pth', map_location='cpu')
        self.linear1.weight.data = classifier_weights['linear1.weight']
        self.linear1.bias.data = classifier_weights['linear1.bias']
        self.linear2.weight.data = classifier_weights['linear2.weight']
        self.linear2.bias.data = classifier_weights['linear2.bias']
        l3_weight = load_l3_weight()
        l3_weight = torch.from_numpy(l3_weight)
        l3_weight = l3_weight.reshape(l3_weight.size(0), -1)
        self.linear3.weight.data = l3_weight.T
        self.linear3.bias.data = torch.zeros(l3_weight.size(1))

    def save_weights(self):
        torch.save(self.state_dict(), align_weights)

    def load_weights(self, map_location='cpu'):
        self.load_state_dict(torch.load(align_weights, map_location=map_location))


@torch.no_grad()
def decode_sample_id(signal_ids_tensor):
    """
    解码样本ID
    输入大小 [batch_size, description_len]
    输出大小 [batch_size]
    """
    signal_ids_tensor = signal_ids_tensor.view(-1, 5) # [batch_size, 5]
    signal_ids_tensor = signal_ids_tensor - signal_token_id
    m_t = torch.tensor(m).unsqueeze(0).to(signal_ids_tensor.device)  # [1, 5]
    signal_ids = signal_ids_tensor * m_t
    return signal_ids.sum(dim=1) # [batch_size]


class IdConverter:
    def __init__(self, train_mode=True):
        self.hp = HyperParameters()
        self.test_file = './cache.npy'
        self.dataset = None
        if train_mode:
            self.dataset = CorpusDataset()

    @torch.no_grad()
    def get_signal(self, signal_ids_tensor, train_mode=True):
        # signal_ids_tensor is tensor [batch_size*description_len]
        res = []
        if train_mode:
            sample_ids = decode_sample_id(signal_ids_tensor)
            for sample_id in sample_ids:
                sample_id, label_id, vib, instruction, response = self.dataset.__getitem__(sample_id)
                res.append(vib)
        else:
            res.append(np.load(self.test_file))
            os.remove(self.test_file)
        data = np.array(res)
        return torch.Tensor(data).to(self.hp.device).detach()

class AlignmentAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.alignment_layer = AlignmentLayer()

    def forward(self, x):
        # x: [batch_size, 2, 24000]
        x = self.feature_encoder(x)
        return self.alignment_layer(x) # [batch_size, description_len, llm_hidden_size]

    def save_weights(self):
        torch.save(self.state_dict(), adapter_weights)

    def load_default(self):
        self.alignment_layer.load_default()
        self.feature_encoder.load_weights(fcn_weights)

    def load_weights(self, map_location='cpu'):
        if not os.path.exists(adapter_weights):
            self.load_default()
            self.save_weights()
        else:
            self.load_state_dict(torch.load(adapter_weights, map_location=map_location))


class ModifiedEmbedding(nn.Module):
    def __init__(self, embedding, train_mode=True):
        super().__init__()
        self.embedding = embedding
        self.adapter = AlignmentAdapter()
        self.adapter.load_weights()
        self.adapter.to(embedding.weight.device)
        self.signal_converter = IdConverter(train_mode=train_mode)

    def forward(self, x):
        if x.max() >= signal_token_id:
            # x: [batch_size, input_len]
            text_part = x[x < signal_token_id].detach() #[batch_size, text_len]
            signal_part = x[x >= signal_token_id].detach() # [batch_size, description_len]
            text_output = self.embedding(text_part) # [batch_size, text_len, llm_hidden_size]
            signal_output = self.signal_converter.get_signal(signal_part, self.training) # [batch_size, 2, 24000]
            signal_output = self.adapter(signal_output) # [batch_size, description_len, llm_hidden_size]
            output = torch.zeros(x.size(0), x.size(1), llm_hidden_size, dtype=torch.bfloat16).to(x.device)
            output[x < signal_token_id] = text_output
            output[x >= signal_token_id] = signal_output.reshape(x.size(0)*5, llm_hidden_size)
            return output
        else:
            return self.embedding(x)


def get_bearllm(train_mode=True):
    hp = HyperParameters()
    config = AutoConfig.from_pretrained(f'{qwen_weights}/config.json')
    model = AutoModelForCausalLM.from_pretrained(
        qwen_weights,
        device_map=hp.device,
        torch_dtype="auto",
        trust_remote_code=True,
        config=config
    )
    embedding = model.get_input_embeddings()
    mod_embedding = ModifiedEmbedding(embedding, train_mode=train_mode)
    model.set_input_embeddings(mod_embedding)
    return model


def mod_xt_for_qwen(xt):
    text_part1 = '<|im_start|>system\n' + sys_prompt + '\n<|im_end|><|im_start|>user\n' + xt.split('#state_place_holder#')[0]
    text_part2 = xt.split('#state_place_holder#')[1] + '<|im_end|>\n<|im_start|>assistant\n'
    return text_part1, text_part2


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True,
                                                padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True,
                                                     padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([x['labels'] for x in batch], batch_first=True,
                                             padding_value=-100)
    return {
        'input_ids': input_ids.detach(),
        'attention_mask': attention_mask.detach(),
        'labels': labels.detach()
    }

def encode_sample_id(x):
    result = []
    remainder = x
    for i in range(len(m)):
        digit = remainder // m[i]
        result.append(digit)
        remainder %= m[i]
    return torch.tensor(result, dtype=torch.int)


class FineTuningDataset(Dataset):
    def __init__(self):
        self.dataset = CorpusDataset()
        self.hp = HyperParameters()
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_weights)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        sample_id, label_id, vib, instruction, response = self.dataset.__getitem__(idx)
        signal_ids = signal_token_id + encode_sample_id(sample_id)
        user_part1, user_part2 = mod_xt_for_qwen(instruction)
        user_part1_ids = self.tokenizer(user_part1, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_part2_ids = self.tokenizer(user_part2, return_tensors='pt', add_special_tokens=False).input_ids[0]
        user_ids = torch.cat([user_part1_ids, signal_ids, user_part2_ids])
        gt_ids = self.tokenizer(response, return_tensors='pt', add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([user_ids, gt_ids, torch.ones(1) * self.tokenizer.pad_token_id])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.cat([torch.ones_like(user_ids) * -100, gt_ids, torch.ones(1) * self.tokenizer.pad_token_id])
        return {
            'input_ids': input_ids.long().detach(),
            'attention_mask': attention_mask.long().detach(),
            'labels': labels.long().detach()
        }


def fine_tuning():
    hp = HyperParameters()
    model = get_bearllm()

    lora_config = LoraConfig(target_modules="all-linear",
                             task_type=TaskType.CAUSAL_LM,  # 任务类型
                             r=hp.r,
                             lora_alpha=hp.lora_alpha,
                             lora_dropout=hp.lora_dropout,
                             )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    dataset = FineTuningDataset()

    train_args = TrainingArguments(
        output_dir=f'{bearllm_weights}/lora',
        per_device_train_batch_size=hp.per_device_train_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        logging_steps=hp.logging_steps,
        num_train_epochs=hp.num_train_epochs,
        save_steps=hp.save_steps,
        learning_rate=hp.learning_rate,
        lr_scheduler_type=hp.lr_scheduler_type,
        
        # 【新增：异步数据加载，防 GPU 饥饿】
        dataloader_num_workers=4,
        
        # 【新增：硬盘防爆机制】限制最多只保留最近的 2 个检查点
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.model.save_pretrained(f'{bearllm_weights}/lora')


if __name__ == "__main__":
    fine_tuning()