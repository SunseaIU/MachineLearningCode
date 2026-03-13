import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 数据预处理：转换为 PyTorch Tensor 并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 PyTorch 张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1,1] 之间
])

# 下载并加载 MNIST 数据集
batch_size = 64

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 检查数据形状
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(f"Batch shape: {images.shape}")  # 输出形状应为 (64, 1, 28, 28)

# 定义 CNN 结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道 1（灰度图），输出通道 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2 最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二个卷积层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层（10 类）

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积 + 池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积 + 池化
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = torch.relu(self.fc1(x))  # 全连接层 1
        x = self.fc2(x)  # 输出层（未经过 softmax，交叉熵损失内部已处理）
        return x

# 实例化模型
model = CNN()
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 训练 CNN
num_epochs = 10  # 训练轮数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU（如果可用）
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 迁移数据到 GPU
        optimizer.zero_grad()  # 清除梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("训练完成！")


# 计算测试集上的准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")


# 获取部分测试样本
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 进行预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 可视化结果
fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    img = images[i].cpu().numpy().squeeze()  # 转换回 NumPy 格式
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"Pred: {predicted[i].item()}")
    axes[i].axis("off")
plt.show()