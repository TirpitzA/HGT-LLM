import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class AlignmentLayer(nn.Module):
    """
    将 64 维特征映射为 Qwen 能理解的多个 Token 词向量。
    """
    def __init__(self, input_dim=64, hidden_dim=3584, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        # 核心映射：将 64 维线性投影到 (num_tokens * hidden_dim) 维
        self.proj = nn.Linear(input_dim, num_tokens * hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, 64)
        out = self.proj(x)
        # 变形为词嵌入标准格式: (batch_size, num_tokens, hidden_dim)
        out = out.view(x.size(0), self.num_tokens, self.hidden_dim)
        return out

class BearingMultimodalQwen(nn.Module):
    def __init__(self, qwen_path, freeze_llm=True, num_vib_tokens=8):
        super().__init__()
        print("[INFO] 正在加载 Qwen 基础模型...")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,  # 推荐使用 bfloat16 或 float16 节省显存
            device_map="auto",
            trust_remote_code=True
        )
        
        # 冻结基础大模型参数
        if freeze_llm:
            for param in self.qwen.parameters():
                param.requires_grad = False
                
        self.hidden_size = self.qwen.config.hidden_size
        self.num_vib_tokens = num_vib_tokens
        
        print(f"[INFO] 初始化 Alignment Layer (64 -> {num_vib_tokens} 个 {self.hidden_size} 维 Tokens)")
        self.alignment_layer = AlignmentLayer(
            input_dim=64, 
            hidden_dim=self.hidden_size, 
            num_tokens=self.num_vib_tokens
        )

    def forward(self, deep_feature, input_ids, attention_mask, labels):
        batch_size = input_ids.size(0)
        
        # 1. 将文本 input_ids 转换为 Embedding
        # shape: (batch_size, seq_len, hidden_size)
        text_embeds = self.qwen.get_input_embeddings()(input_ids)
        
        # 2. 将振动特征转换为 Embedding
        # shape: (batch_size, num_vib_tokens, hidden_size)
        vib_embeds = self.alignment_layer(deep_feature).to(text_embeds.dtype)
        
        # 3. 模态拼接：将振动特征放在序列最开头
        # shape: (batch_size, num_vib_tokens + seq_len, hidden_size)
        inputs_embeds = torch.cat([vib_embeds, text_embeds], dim=1)
        
        # 4. 动态扩展 Attention Mask（补齐前面的振动 Token 位置）
        vib_mask = torch.ones(
            (batch_size, self.num_vib_tokens), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        extended_attention_mask = torch.cat([vib_mask, attention_mask], dim=1)
        
        # 5. 动态扩展 Labels（振动 Token 不参与 Loss 计算，填 -100）
        vib_labels = torch.full(
            (batch_size, self.num_vib_tokens), 
            -100, 
            dtype=labels.dtype, 
            device=labels.device
        )
        extended_labels = torch.cat([vib_labels, labels], dim=1)
        
        # 6. 送入 Qwen 计算
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )
        
        return outputs