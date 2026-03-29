import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# =========================
# 1. 固定随机种子，保证结果可复现
# =========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)


# =========================
# 2. 手写 Softmax
#    不能直接调用 nn.Softmax 或 torch.softmax
# =========================
def manual_softmax(logits):
    """
    logits: [batch_size, num_classes]
    return: [batch_size, num_classes]
    """
    # 为了数值稳定，先减去每行最大值
    max_vals, _ = torch.max(logits, dim=1, keepdim=True)
    exp_vals = torch.exp(logits - max_vals)
    probs = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
    return probs


# =========================
# 3. 手写交叉熵损失
#    不直接用 nn.CrossEntropyLoss
# =========================
def manual_cross_entropy(probs, labels):
    """
    probs:  [batch_size, num_classes]
    labels: [batch_size]
    """
    eps = 1e-9  # 防止 log(0)
    correct_class_probs = probs[torch.arange(labels.size(0)), labels]
    loss = -torch.log(correct_class_probs + eps).mean()
    return loss


# =========================
# 4. 定义一个隐藏层的 MLP
# =========================
class IrisMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super(IrisMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)       # 原始分数
        probs = manual_softmax(logits)  # 手写 softmax
        return logits, probs


# =========================
# 5. 加载 Iris 数据集
# =========================
iris = load_iris()
X = iris.data          # [150, 4]
y = iris.target        # [150]

# 训练集 / 测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 标准化（提高训练稳定性）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转成 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# =========================
# 6. 初始化模型、优化器
# =========================
model = IrisMLP(input_dim=4, hidden_dim=16, output_dim=3)
optimizer = optim.SGD(model.parameters(), lr=0.05)


# =========================
# 7. 训练循环
# =========================
epochs = 200

for epoch in range(epochs):
    model.train()

    # 前向传播
    logits, probs = model(X_train)

    # 计算损失
    loss = manual_cross_entropy(probs, y_train)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新参数
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        # 训练集准确率
        preds = torch.argmax(probs, dim=1)
        acc = (preds == y_train).float().mean().item()
        print(f"Epoch [{epoch+1:3d}/{epochs}]  Loss: {loss.item():.4f}  Train Acc: {acc:.4f}")


# =========================
# 8. 测试集评估
# =========================
model.eval()
with torch.no_grad():
    _, test_probs = model(X_test)
    test_preds = torch.argmax(test_probs, dim=1)
    test_acc = (test_preds == y_test).float().mean().item()

print("\n测试集预测结果：", test_preds.numpy())
print("测试集真实标签：", y_test.numpy())
print(f"测试集准确率 Accuracy: {test_acc:.4f}")