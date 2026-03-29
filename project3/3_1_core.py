# problem3_1_gdn_core.py
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaRule(nn.Module):
    """
    题目 3-1：GatedDeltaNet 核心算子基础实现（递归版）
    这里只实现最核心的状态更新逻辑：
        S_t = S_{t-1} ( alpha_t ( I - beta_t k_t k_t^T ) ) + beta_t v_t k_t^T

    说明：
    1. 使用 nn.Linear 等中级封装生成 q, k, v, alpha, beta, gate
    2. 初始版本使用纯循环（recurrent）方式，优先保证逻辑正确
    3. 输入输出均按 batch-first 组织：x.shape = (B, T, D)
    """

    def __init__(self, dim, state_dim):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # 输入映射
        self.q_proj = nn.Linear(dim, state_dim)
        self.k_proj = nn.Linear(dim, state_dim)
        self.v_proj = nn.Linear(dim, state_dim)

        # 标量门控参数
        self.alpha_proj = nn.Linear(dim, 1)
        self.beta_proj = nn.Linear(dim, 1)
        self.gate_proj = nn.Linear(dim, state_dim)

        # 输出投影
        self.out_proj = nn.Linear(state_dim, dim)

    def forward(self, x, return_state=False):
        """
        x: (B, T, D)
        return:
            y: (B, T, D)
            可选返回最终状态 S: (B, C, C)，这里 C = state_dim
        """
        B, T, D = x.shape
        C = self.state_dim
        device = x.device
        dtype = x.dtype

        # 映射得到各项
        q = self.q_proj(x)                      # (B, T, C)
        k = self.k_proj(x)                      # (B, T, C)
        v = self.v_proj(x)                      # (B, T, C)

        # 为了数值更稳定，对 k 做 L2 归一化
        k = F.normalize(k, p=2, dim=-1)

        # alpha, beta 压到 (0, 1)
        alpha = torch.sigmoid(self.alpha_proj(x))   # (B, T, 1)
        beta = torch.sigmoid(self.beta_proj(x))     # (B, T, 1)

        # 输出门
        gate = torch.sigmoid(self.gate_proj(x))     # (B, T, C)

        # 初始隐状态 S_0
        # S 视作一个矩阵状态，shape = (B, C, C)
        S = torch.zeros(B, C, C, device=device, dtype=dtype)

        outputs = []
        I = torch.eye(C, device=device, dtype=dtype).unsqueeze(0)  # (1, C, C)

        for t in range(T):
            k_t = k[:, t, :]   # (B, C)
            v_t = v[:, t, :]   # (B, C)
            q_t = q[:, t, :]   # (B, C)
            a_t = alpha[:, t, :].unsqueeze(-1)  # (B, 1, 1)
            b_t = beta[:, t, :].unsqueeze(-1)   # (B, 1, 1)
            g_t = gate[:, t, :]                 # (B, C)

            # k_t k_t^T -> (B, C, C)
            kkT = torch.bmm(k_t.unsqueeze(-1), k_t.unsqueeze(1))

            # v_t k_t^T -> (B, C, C)
            vkT = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))

            # 按题目给出的递推规则更新
            # S_t = S_{t-1} * [ alpha_t (I - beta_t * k_t k_t^T) ] + beta_t * v_t k_t^T
            transition = a_t * (I - b_t * kkT)
            S = torch.bmm(S, transition) + b_t * vkT

            # 输出 o_t = q_t * S_t
            # q_t: (B, C), S: (B, C, C) -> (B, C)
            o_t = torch.bmm(q_t.unsqueeze(1), S).squeeze(1)

            # 加输出门
            o_t = o_t * g_t

            # 输出投影回 dim
            y_t = self.out_proj(o_t)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, D)

        if return_state:
            return y, S
        return y


if __name__ == "__main__":
    # 简单自测
    torch.manual_seed(42)
    x = torch.randn(2, 8, 64)  # batch=2, seq_len=8, dim=64

    model = GatedDeltaRule(dim=64, state_dim=32)
    y, final_state = model(x, return_state=True)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
    print("最终状态形状:", final_state.shape)