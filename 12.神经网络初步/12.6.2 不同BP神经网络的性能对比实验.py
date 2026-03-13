import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# 生成一些简单数据：分类任务
def generate_data(n=100):
    X = torch.randn(n, 2)
    y = (X[:, 0] * X[:, 1] > 0).long()  # label: 正负号是否一致
    return X, y

# 可选择的激活函数
def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation")

# 可选择的损失函数
def get_loss(name):
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss")

# 简单的 2 层前馈神经网络
class SimpleNet(nn.Module):
    def __init__(self, activation_name):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.activation = get_activation(activation_name)
        self.fc2 = nn.Linear(10, 2)  # 二分类任务输出为2个节点

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# 训练函数
def train_model(activation_name, loss_name, epochs=100):
    print(f"\n🔧 训练中：激活函数 = {activation_name}, 损失函数 = {loss_name}")
    X, y = generate_data(200)
    model = SimpleNet(activation_name)
    criterion = get_loss(loss_name)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)

        if loss_name == 'mse':
            # 将 y 转为 one-hot 形式用于 MSE
            y_onehot = F.one_hot(y, num_classes=2).float()
            loss = criterion(outputs, y_onehot)
        else:
            loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            pred = torch.argmax(outputs, dim=1)
            acc = (pred == y).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

# 测试不同组合
activations = ['relu', 'sigmoid', 'tanh']
losses = ['mse', 'cross_entropy']
for act in activations:
    for loss in losses:
        train_model(act, loss)