# problem3_2_train_fmnist_gdn.py
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class GatedDeltaRule(nn.Module):
    """
    递归版 GDN 核心算子
    """
    def __init__(self, dim, state_dim):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        self.q_proj = nn.Linear(dim, state_dim)
        self.k_proj = nn.Linear(dim, state_dim)
        self.v_proj = nn.Linear(dim, state_dim)

        self.alpha_proj = nn.Linear(dim, 1)
        self.beta_proj = nn.Linear(dim, 1)
        self.gate_proj = nn.Linear(dim, state_dim)

        self.out_proj = nn.Linear(state_dim, dim)

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
        I = torch.eye(C, device=device, dtype=dtype).unsqueeze(0)

        outputs = []
        for t in range(T):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]

            a_t = alpha[:, t, :].unsqueeze(-1)
            b_t = beta[:, t, :].unsqueeze(-1)
            g_t = gate[:, t, :]

            kkT = torch.bmm(k_t.unsqueeze(-1), k_t.unsqueeze(1))
            vkT = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))

            transition = a_t * (I - b_t * kkT)
            S = torch.bmm(S, transition) + b_t * vkT

            o_t = torch.bmm(q_t.unsqueeze(1), S).squeeze(1)
            o_t = o_t * g_t
            y_t = self.out_proj(o_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class GDNBlock(nn.Module):
    def __init__(self, dim, state_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gdn = GatedDeltaRule(dim, state_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.gdn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FashionMNISTGDNClassifier(nn.Module):
    """
    将 28x28 图像转成长度为 28 的序列，每个 token 的维度也是 28
    再映射到 embed_dim 后输入 GDN 堆栈
    """
    def __init__(self, seq_len=28, token_dim=28, embed_dim=128, state_dim=64,
                 depth=4, num_classes=10, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.token_dim = token_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(token_dim, embed_dim)

        self.blocks = nn.ModuleList([
            GDNBlock(embed_dim, state_dim, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 1, 28, 28)
        这里采用“按行展开”为序列：
            序列长度 T=28
            每个 token 维度=28
        """
        x = x.squeeze(1)           # (B, 28, 28)
        x = self.input_proj(x)     # (B, 28, embed_dim)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 分类头：对序列做平均池化
        x = x.mean(dim=1)          # (B, embed_dim)
        logits = self.head(x)      # (B, 10)
        return logits


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "acc_curve.png"))
    plt.close()


def train():
    # --------------------------
    # 超参数
    # --------------------------
    batch_size = 128
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4

    embed_dim = 128
    state_dim = 64
    depth = 4
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)

    # --------------------------
    # 数据集
    # --------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # --------------------------
    # 模型
    # --------------------------
    model = FashionMNISTGDNClassifier(
        seq_len=28,
        token_dim=28,
        embed_dim=embed_dim,
        state_dim=state_dim,
        depth=depth,
        num_classes=10,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------------------
    # 训练记录
    # --------------------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0
    os.makedirs("outputs", exist_ok=True)

    # --------------------------
    # 开始训练
    # --------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "outputs/best_gdn_fmnist.pth")

    print(f"\n最佳测试准确率: {best_acc:.4f}")
    plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir="outputs")
    print("训练完成，模型与曲线已保存到 outputs/ 目录。")


if __name__ == "__main__":
    train()