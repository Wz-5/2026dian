import math
import torch
import torch.nn as nn


class StandardMHAWithKVCache(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, H, L, Hd]
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, Hd] -> [B, L, D]
        B, H, L, Hd = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, L, H * Hd)
        return x

    def forward(self, x: torch.Tensor, past_key_values=None):
        """
        x: [B, L, D]
        past_key_values:
            None
            or (past_k, past_v)
            past_k: [B, H, L_past, Hd]
            past_v: [B, H, L_past, Hd]
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)  # [B, H, Lq, Hd]
        k = self._split_heads(k)  # [B, H, Lk_new, Hd]
        v = self._split_heads(v)  # [B, H, Lv_new, Hd]

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)  # 在序列长度维度拼接
            v = torch.cat([past_v, v], dim=2)

        new_past_key_values = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        out = self._merge_heads(context)
        out = self.out_proj(out)

        return out, new_past_key_values, q.shape, k.shape, v.shape


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    init_seq_len = 10
    hidden_dim = 32
    num_heads = 4

    mha = StandardMHAWithKVCache(hidden_dim=hidden_dim, num_heads=num_heads)

    # 先输入长度为10的初始序列
    x_init = torch.randn(batch_size, init_seq_len, hidden_dim)
    out, past_key_values, q_shape, k_shape, v_shape = mha(x_init, past_key_values=None)

    print("=== 初始输入阶段 ===")
    print("Q shape:", q_shape)  # [B, H, 10, Hd]
    print("K shape:", k_shape)  # [B, H, 10, Hd]
    print("V shape:", v_shape)  # [B, H, 10, Hd]
    print()

    # 再循环5次，每次输入一个新token
    print("=== 自回归解码阶段 ===")
    for step in range(1, 6):
        x_new = torch.randn(batch_size, 1, hidden_dim)  # 每次只输入1个token
        out, past_key_values, q_shape, k_shape, v_shape = mha(x_new, past_key_values=past_key_values)

        print(f"Step {step}:")
        print("当前输入 token 形状:", x_new.shape)
        print("Q shape:", q_shape)   # 这里序列长度应始终为1
        print("K shape:", k_shape)   # 序列长度从10逐步到15
        print("V shape:", v_shape)
        print("KV Cache总长度:", k_shape[2])
        print("-" * 50)