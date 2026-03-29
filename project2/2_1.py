import math
import torch
import torch.nn as nn


class StandardMHA(nn.Module):
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
        # x: [B, L, D]
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B, H, L, Hd]
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, Hd]
        B, H, L, Hd = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, L, H, Hd]
        x = x.view(B, L, H * Hd)  # [B, L, D]
        return x

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, L, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)  # [B, H, L, Hd]
        k = self._split_heads(k)  # [B, H, L, Hd]
        v = self._split_heads(v)  # [B, H, L, Hd]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, L, L]
        context = torch.matmul(attn_weights, v)  # [B, H, L, Hd]

        out = self._merge_heads(context)  # [B, L, D]
        out = self.out_proj(out)  # [B, L, D]

        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    hidden_dim = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, hidden_dim)

    mha = StandardMHA(hidden_dim=hidden_dim, num_heads=num_heads)
    out = mha(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("形状是否一致:", x.shape == out.shape)