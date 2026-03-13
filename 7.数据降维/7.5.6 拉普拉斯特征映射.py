import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（SimHei 支持中文）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 生成瑞士卷数据
n_samples = 1000
X, color = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=42)
# 构建k近邻邻接矩阵
n_neighbors = 10
W = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
W = 0.5 * (W + W.T)  # 对称化邻接矩阵
W = W.toarray()
# 计算度矩阵和拉普拉斯矩阵
D = np.diag(W.sum(axis=1))
L = D - W
# 求解广义特征值问题 Lv = λDv
eigen_values, eigen_vectors = eigh(L, D)
# 按特征值升序排序，跳过第一个特征向量（全1）
sorted_indices = np.argsort(eigen_values)
embedding = eigen_vectors[:, sorted_indices][:, 1:3]  # 取第二和第三小的特征向量
# 可视化结果（灰度图）
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='gray', alpha=0.6)
plt.title('瑞士卷的拉普拉斯特征映射')
plt.xlabel('成分 1')
plt.ylabel('成分 2')
plt.colorbar(label='原始高度')
plt.show()