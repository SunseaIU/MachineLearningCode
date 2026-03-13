import torch
import numpy as np
# NumPy实现
def dropout_forward(x, dropout_rate, training=True):
    if not training or dropout_rate == 0.0:
        return x, None
    mask = (np.random.rand(*x.shape) > dropout_rate).astype(np.float32)
    out = x * mask / (1.0 - dropout_rate)  # 缩放以保持期望值不变
    return out, mask

def dropout_backward(dout, mask, dropout_rate):
    if mask is None:
        return dout
    return dout * mask / (1.0 - dropout_rate)

x = np.array([[1.0, 2.0, 3.0]])
out, mask = dropout_forward(x, dropout_rate=0.5, training=True)
print("Dropped output:", out)

# 假设反向传播梯度是全1
dx = dropout_backward(np.ones_like(x), mask, 0.5)
print("Gradient after dropout:", dx)


# PyTorch实现：
import torch
import torch.nn as nn

# 方法一：直接使用 PyTorch 的 Dropout 层
dropout = nn.Dropout(p=0.5)

x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
dropout.train()  # 设置为训练模式
out = dropout(x)
print("Dropped output:", out)

# 方法二：自定义 Dropout 函数
def custom_dropout(x, dropout_rate, training=True):
    if not training or dropout_rate == 0.0:
        return x
    mask = (torch.
            rand_like(x) > dropout_rate).float()
    return x * mask / (1.0 - dropout_rate)
#x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
#out = custom_dropout(x, dropout_rate=0.5, training=True)
#print("Custom Dropped output:", out)