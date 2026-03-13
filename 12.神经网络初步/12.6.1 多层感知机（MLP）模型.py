import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义一个多层感知机（MLP）类
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 定义三层结构：输入层、隐藏层、输出层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 隐藏层的激活函数
        x = self.fc3(x)  # 输出层
        return x

# 创建一个简单的训练函数
def train_mlp(input_size, hidden_size, output_size, epochs=100, lr=0.15):
    # 创建随机数据：100个样本，输入维度为2，输出维度为1（分类任务）
    #加入噪声，使数据更加复杂
    X = torch.randn(100, input_size) + 0.1 * torch.randn(100, input_size)
    y = (X[:, 0] * X[:, 1] > 0).long()  # 根据特定规则生成标签

    # 初始化模型、损失函数和优化器
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()  # 分类任务的损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 训练过程
    for epoch in range(epochs):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(X)  # 前向传播
        loss = criterion(outputs, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 每20个epoch输出一次训练信息
        if epoch % 20 == 0 or epoch == epochs - 1:
            pred = torch.argmax(outputs, dim=1)  # 获取预测结果
            acc = (pred == y).float().mean().item()  # 计算准确率
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

# 调用训练函数，定义输入、隐藏层、输出的尺寸
train_mlp(input_size=2, hidden_size=10, output_size=2)