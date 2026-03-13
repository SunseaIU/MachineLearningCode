import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# ===== 模拟二分类数据（可换成 MNIST） =====
X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

# ===== MLP模型定义 =====
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ===== 多样性策略参数 =====
num_learners = 5
hidden_dim_options = [8, 16, 32, 64]
dropout_options = [0.0, 0.2, 0.5]
learning_rates = [0.01, 0.001]
random_seeds = list(range(num_learners))

# ===== 创建多样性模型（集成） =====
learners = []
optimizers = []

for i in range(num_learners):
    torch.manual_seed(random_seeds[i])

    # 输入属性扰动：随机丢弃特征维度
    feature_mask = torch.rand(train_x.shape[1]) > 0.5
    train_x_masked = train_x.clone()
    train_x_masked[:, ~feature_mask] = 0.0

    # 输出扰动：对一部分样本加标签噪声
    train_y_noisy = train_y.clone()
    noise_idx = torch.rand(len(train_y)) < 0.1
    train_y_noisy[noise_idx] = 1 - train_y_noisy[noise_idx]  # 二分类翻转标签

    # 随机扰动参数
    hidden_dim = random.choice(hidden_dim_options)
    dropout_rate = random.choice(dropout_options)
    lr = random.choice(learning_rates)

    # 初始化模型与优化器
    model = MLP(input_dim=train_x.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 简单训练
    for epoch in range(10):
        model.train()
        logits = model(train_x_masked)
        loss = F.cross_entropy(logits, train_y_noisy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    learners.append(model)

# ===== 集成预测（投票） =====
def ensemble_predict(models, x):
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(x)
            preds.append(torch.argmax(out, dim=1))
    preds = torch.stack(preds)
    final = torch.mode(preds, dim=0)[0]  # 多数投票
    return final

# ===== 测试准确率 =====
pred = ensemble_predict(learners, test_x)
acc = (pred == test_y).float().mean()
print("集成模型准确率：", acc.item())