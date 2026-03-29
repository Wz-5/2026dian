# problem3_3_chunkwise.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ChunkwiseGatedDeltaRule(nn.Module):
    """
    3-3 方法二：
    Chunkwise 版本的 GDN 核心算子

    思路：
    1. 先把序列切成多个 chunk
    2. 每个 chunk 内仍做递归更新
    3. chunk 与 chunk 之间传递状态
    4. 相比完全逐 token Python 循环，更利于后续进一步并行优化和吞吐改造

    这是一个适合作业/实验报告展示的“块状并行”基线实现。
    """
    def __init__(self, dim, state_dim, chunk_size=7):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(dim, state_dim)
        self.k_proj = nn.Linear(dim, state_dim)
        self.v_proj = nn.Linear(dim, state_dim)

        self.alpha_proj = nn.Linear(dim, 1)
        self.beta_proj = nn.Linear(dim, 1)
        self.gate_proj = nn.Linear(dim, state_dim)

        self.out_proj = nn.Linear(state_dim, dim)

    def _run_chunk(self, q, k, v, alpha, beta, gate, S):
        """
        对单个 chunk 做递归
        输入:
            q,k,v: (B, Tc, C)
            alpha,beta: (B, Tc, 1)
            gate: (B, Tc, C)
            S: (B, C, C) 上一个 chunk 结束后的状态
        返回:
            y_chunk: (B, Tc, D)
            S: 更新后的状态
        """
        B, Tc, C = q.shape
        device = q.device
        dtype = q.dtype
        I = torch.eye(C, device=device, dtype=dtype).unsqueeze(0)

        outs = []
        for t in range(Tc):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            a_t = alpha[:, t, :].unsqueeze(-1)
            b_t = beta[:, t, :].unsqueeze(-1)
            g_t = gate[:, t, :]

            kkT = torch.bmm(k_t.unsqueeze(-1), k_t.unsqueeze(1))
            vkT = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))

            S = torch.bmm(S, a_t * (I - b_t * kkT)) + b_t * vkT
            o_t = torch.bmm(q_t.unsqueeze(1), S).squeeze(1)
            o_t = o_t * g_t
            y_t = self.out_proj(o_t)
            outs.append(y_t)

        return torch.stack(outs, dim=1), S

    def forward(self, x):
        B, T, D = x.shape
        C = self.state_dim
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x)
        k = F.normalize(self.k_proj(x), p=2, dim=-1)
        v = self.v_proj(x)
        alpha = torch.sigmoid(self.alpha_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))

        S = torch.zeros(B, C, C, device=device, dtype=dtype)
        outputs = []

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)

            y_chunk, S = self._run_chunk(
                q[:, start:end, :],
                k[:, start:end, :],
                v[:, start:end, :],
                alpha[:, start:end, :],
                beta[:, start:end, :],
                gate[:, start:end, :],
                S
            )
            outputs.append(y_chunk)

        return torch.cat(outputs, dim=1)


class GDNBlockChunkwise(nn.Module):
    def __init__(self, dim, state_dim, chunk_size=7, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gdn = ChunkwiseGatedDeltaRule(dim, state_dim, chunk_size=chunk_size)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.gdn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FashionMNISTGDNChunkwise(nn.Module):
    def __init__(self, seq_len=28, token_dim=28, embed_dim=128,
                 state_dim=64, depth=4, chunk_size=7, num_classes=10, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(token_dim, embed_dim)

        self.blocks = nn.ModuleList([
            GDNBlockChunkwise(embed_dim, state_dim, chunk_size=chunk_size,
                              mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)         # (B, 28, 28)
        x = self.input_proj(x)   # (B, 28, embed_dim)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def evaluate(model, dataloader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train():
    batch_size = 128
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4
    chunk_size = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FashionMNISTGDNChunkwise(
        seq_len=28,
        token_dim=28,
        embed_dim=128,
        state_dim=64,
        depth=4,
        chunk_size=chunk_size,
        num_classes=10,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    best_acc = 0.0
    os.makedirs("outputs_chunkwise", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        test_loss, test_acc = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "outputs_chunkwise/best_model.pth")

    print("best test acc =", best_acc)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(test_losses, label="test_loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs_chunkwise/loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="train_acc")
    plt.plot(test_accs, label="test_acc")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs_chunkwise/acc_curve.png")
    plt.close()


if __name__ == "__main__":
    train()