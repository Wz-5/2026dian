
# 3_3_pos_bias.py
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class GatedDeltaRule(nn.Module):
    def __init__(self, dim, state_dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, state_dim)
        self.k_proj = nn.Linear(dim, state_dim)
        self.v_proj = nn.Linear(dim, state_dim)
        self.alpha_proj = nn.Linear(dim, 1)
        self.beta_proj = nn.Linear(dim, 1)
        self.gate_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim)
        self.state_dim = state_dim

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

        outs = []
        for t in range(T):
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

            outs.append(self.out_proj(o_t))

        return torch.stack(outs, dim=1)


class GDNBlock(nn.Module):
    def __init__(self, dim, state_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gdn = GatedDeltaRule(dim, state_dim)

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


class FashionMNISTGDNWithPosBias(nn.Module):
    def __init__(self, seq_len=28, token_dim=28, embed_dim=128,
                 state_dim=64, depth=4, num_classes=10, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(token_dim, embed_dim)

        # 可学习位置偏置
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            GDNBlock(embed_dim, state_dim, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)             # (B, 28, 28)
        x = self.input_proj(x)       # (B, 28, embed_dim)
        x = x + self.pos_embed       # 加位置偏置

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
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
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # loss 图
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # train_val 图（准确率）
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Val Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "train_val_acc_curve.png"))
    plt.close()


def train():
    batch_size = 128
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4
    val_ratio = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_set = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_size = int(len(full_train_set) * (1 - val_ratio))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FashionMNISTGDNWithPosBias(
        seq_len=28, token_dim=28, embed_dim=128,
        state_dim=64, depth=4, num_classes=10, dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    save_dir = "outputs_pos_bias"
    os.makedirs(save_dir, exist_ok=True)

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
        val_loss, val_acc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    print(f"\nBest val acc = {best_val_acc:.4f}")

    # 绘图
    plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir)

    # 在 test 集上做最终评估
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Final test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}")


if __name__ == "__main__":
    train()