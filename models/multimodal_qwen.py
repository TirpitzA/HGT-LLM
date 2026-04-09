"""
multimodal_qwen.py (Token Replacement 版)
==========================================
核心改动：与 BearLLM 对齐，采用底层 Token 替换而非显式 Embedding 拼接。
当模型在 input_ids 中看到 signal_token_id 占位符时，ModifiedEmbedding 会将其替换为
由 AlignmentLayer 投射出的振动语义向量。

关键参数：
  - SIGNAL_TOKEN_ID = 151925 (与 BearLLM 保持一致)
  - NUM_VIB_TOKENS  = 8     (HGT-LLM 使用 8 个振动 Token)
  - hidden_size     = 1536  (Qwen2-1.5B)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ── 全局常量 (与 BearLLM .env 对齐) ──────────────────────────────────────────
SIGNAL_TOKEN_ID = 151925
DEFAULT_NUM_VIB_TOKENS = 8

class AlignmentLayer(nn.Module):
    """
    将 64 维 deep_feature 映射为 LLM 可理解的多个 Token 词向量。
    输入: (batch_size, 64)
    输出: (batch_size, num_tokens, hidden_dim)
    """
    def __init__(self, input_dim=64, hidden_dim=1536, num_tokens=DEFAULT_NUM_VIB_TOKENS):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(input_dim, num_tokens * hidden_dim)

    def forward(self, x):
        out = self.proj(x)
        return out.view(x.size(0), self.num_tokens, self.hidden_dim)


class ModifiedEmbedding(nn.Module):
    """
    与 BearLLM 对齐的底层 Token 替换 Embedding。
    当 input_ids 中出现 >= signal_token_id 的 Token 时，
    将其替换为 AlignmentLayer 产出的振动语义向量。
    """
    def __init__(self, original_embedding, alignment_layer,
                 signal_token_id=SIGNAL_TOKEN_ID, num_vib_tokens=DEFAULT_NUM_VIB_TOKENS):
        super().__init__()
        self.embedding = original_embedding
        self.alignment_layer = alignment_layer
        self.signal_token_id = signal_token_id
        self.num_vib_tokens = num_vib_tokens
        self.hidden_size = original_embedding.weight.shape[1]

        # 振动特征缓存：由外部在每次 forward 前通过 set_feature() 注入
        self._cached_feature = None

    def set_feature(self, deep_feature):
        """在每次前向传播前，由训练/推理循环调用，设置当前 batch 的深层特征"""
        self._cached_feature = deep_feature

    def forward(self, x):
        has_signal = (x >= self.signal_token_id).any()

        if has_signal and self._cached_feature is not None:
            batch_size, seq_len = x.shape
            signal_mask = (x >= self.signal_token_id)
            text_mask = ~signal_mask

            # 1. 正常 Embedding：将信号占位符位置替零以避免 OOV，然后查表
            safe_ids = x.clone()
            safe_ids[signal_mask] = 0  # 占位符位置填 0，防止越界
            text_embeds = self.embedding(safe_ids)

            # 2. 振动特征 Embedding：通过 AlignmentLayer 投射
            vib_embeds = self.alignment_layer(self._cached_feature)  # (B, num_vib_tokens, H)
            vib_embeds = vib_embeds.to(text_embeds.dtype)

            # 3. Token 替换：将信号占位符位置的 Embedding 替换为振动向量
            output = text_embeds.clone()
            output[signal_mask] = vib_embeds.reshape(-1, self.hidden_size)

            return output
        else:
            return self.embedding(x)


class BearingMultimodalQwen(nn.Module):
    """
    HGT-LLM 多模态包装器 (Token Replacement 版)。
    与旧版的接口差异：
      - forward() 仍然接受 deep_feature, input_ids, attention_mask, labels
      - 但内部不再做序列拼接，而是通过 ModifiedEmbedding 进行 Token 替换
      - input_ids 中必须包含 SIGNAL_TOKEN_ID 占位符 (由 Dataset 构造)
    """
    def __init__(self, qwen_path, freeze_llm=True,
                 num_vib_tokens=DEFAULT_NUM_VIB_TOKENS,
                 signal_token_id=SIGNAL_TOKEN_ID):
        super().__init__()
        self.signal_token_id = signal_token_id
        self.num_vib_tokens = num_vib_tokens

        print("[INFO] 正在加载 Qwen 基础模型...")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if freeze_llm:
            for param in self.qwen.parameters():
                param.requires_grad = False

        self.hidden_size = self.qwen.config.hidden_size

        print(f"[INFO] 初始化 AlignmentLayer (64 -> {num_vib_tokens} × {self.hidden_size})")
        self.alignment_layer = AlignmentLayer(
            input_dim=64,
            hidden_dim=self.hidden_size,
            num_tokens=num_vib_tokens
        )

        # 【核心】替换 Embedding 层为 ModifiedEmbedding
        original_embedding = self.qwen.get_input_embeddings()
        self.modified_embedding = ModifiedEmbedding(
            original_embedding, self.alignment_layer,
            signal_token_id, num_vib_tokens
        )
        self.qwen.set_input_embeddings(self.modified_embedding)

        print(f"[INFO] ModifiedEmbedding 注入完毕 (signal_token_id={signal_token_id})")

    def forward(self, deep_feature, input_ids, attention_mask, labels):
        """
        前向传播。
        与旧版保持相同的函数签名，但内部改用 Token 替换。
        input_ids 中必须包含 num_vib_tokens 个 signal_token_id 占位符。
        """
        # 注入当前 batch 的振动特征到 Embedding 缓存
        self.modified_embedding.set_feature(deep_feature)

        # 直接传入 input_ids，ModifiedEmbedding 会自动处理占位符替换
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs