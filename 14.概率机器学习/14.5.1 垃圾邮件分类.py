import torch
import numpy as np

# 构造简单数据集
X_train = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]])
y_train = np.array([1, 1, 0, 0])  # 1: spam, 0: normal

# 统计先验概率
p_y = np.mean(y_train)

# 条件概率估计
alpha = 1  # 拉普拉斯平滑
p_x_given_y = np.zeros((2, 3))
for y in [0, 1]:
    X_y = X_train[y_train == y]
    p_x_given_y[y] = (np.sum(X_y, axis=0) + alpha) / (len(X_y) + 2)

# 分类函数
def predict(x):
    p_spam = p_y * np.prod(p_x_given_y[1] ** x * (1 - p_x_given_y[1]) ** (1 - x))
    p_normal = (1 - p_y) * np.prod(p_x_given_y[0] ** x * (1 - p_x_given_y[0]) ** (1 - x))
    return 1 if p_spam > p_normal else 0

# 测试
x_new = np.array([1, 0, 0])
print("Predicted label:", predict(x_new))