import math
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()

        assert hidden_dim % num_q_heads == 0, "hidden_dim 必须能被 num_q_heads 整除"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须能被 num_kv_heads 整除"

        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, hidden_dim)

    def _split_q_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, num_q_heads * Hd] -> [B, num_q_heads, L, Hd]
        B, L, _ = x.shape
        x = x.view(B, L, self.num_q_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _split_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, num_kv_heads * Hd] -> [B, num_kv_heads, L, Hd]
        B, L, _ = x.shape
        x = x.view(B, L, self.num_kv_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_q_heads, L, Hd] -> [B, L, num_q_heads * Hd]
        B, H, L, Hd = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, L, H * Hd)
        return x

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 KV 头复制到与 Q 头数量一致
        x: [B, num_kv_heads, L, Hd]
        return: [B, num_q_heads, L, Hd]
        """
        B, H_kv, L, Hd = x.shape
        x = x.unsqueeze(2)  # [B, H_kv, 1, L, Hd]
        x = x.repeat(1, 1, self.group_size, 1, 1)  # [B, H_kv, group_size, L, Hd]
        x = x.view(B, self.num_q_heads, L, Hd)  # [B, num_q_heads, L, Hd]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        q = self.q_proj(x)  # [B, L, num_q_heads * Hd]
        k = self.k_proj(x)  # [B, L, num_kv_heads * Hd]
        v = self.v_proj(x)  # [B, L, num_kv_heads * Hd]

        q = self._split_q_heads(q)    # [B, num_q_heads, L, Hd]
        k = self._split_kv_heads(k)   # [B, num_kv_heads, L, Hd]
        v = self._split_kv_heads(v)   # [B, num_kv_heads, L, Hd]

        # 将KV扩展到和Q头数一致
        k = self._repeat_kv(k)        # [B, num_q_heads, L, Hd]
        v = self._repeat_kv(v)        # [B, num_q_heads, L, Hd]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        out = self._merge_heads(context)
        out = self.out_proj(out)

        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6
    hidden_dim = 32
    num_q_heads = 8

    x = torch.randn(batch_size, seq_len, hidden_dim)

    print("输入形状:", x.shape)
    print("=" * 60)

    # 1. MHA: num_kv_heads == num_q_heads
    mha = GroupedQueryAttention(hidden_dim=hidden_dim, num_q_heads=num_q_heads, num_kv_heads=8)
    out_mha = mha(x)
    print("MHA 模式:")
    print("num_q_heads = 8, num_kv_heads = 8")
    print("输出形状:", out_mha.shape)
    print("-" * 60)

    # 2. GQA: 1 < num_kv_heads < num_q_heads
    gqa = GroupedQueryAttention(hidden_dim=hidden_dim, num_q_heads=num_q_heads, num_kv_heads=4)
    out_gqa = gqa(x)
    print("GQA 模式:")
    print("num_q_heads = 8, num_kv_heads = 4")
    print("输出形状:", out_gqa.shape)
    print("-" * 60)

    # 3. MQA: num_kv_heads == 1
    mqa = GroupedQueryAttention(hidden_dim=hidden_dim, num_q_heads=num_q_heads, num_kv_heads=1)
    out_mqa = mqa(x)
    print("MQA 模式:")
    print("num_q_heads = 8, num_kv_heads = 1")
    print("输出形状:", out_mqa.shape)
    print("-" * 60)