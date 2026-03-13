import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体，防止报错
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 加载手写数字数据集
digits = datasets.load_digits()
X, y = digits.data, digits.target  # X 是特征，y 是标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算每个点的k近邻
k = 10
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)

# 为每个点计算局部切空间，使用PCA
def compute_local_tangent_space(X, k):
    n_samples, n_features = X.shape
    local_bases = np.zeros((n_samples, k-1, n_features))
    for i in range(n_samples):
        distances, indices = neighbors.kneighbors(X[i].reshape(1, -1))
        neighbors_data = X[indices[0]]
        pca = PCA(n_components=k-1)
        pca.fit(neighbors_data)
        local_bases[i] = pca.components_
    return local_bases

# 计算局部切空间
local_bases = compute_local_tangent_space(X_scaled, k)

# 显示原始数据中的一些手写数字样本
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"数字 {y[i]}")
    ax.axis('off')
plt.show()

# 使用PCA进行降维 (简化版LSTA)
pca = PCA(n_components=2)
X_lsta = pca.fit_transform(X_scaled)

# 定义不同数字的符号，适合黑白打印
markers = ['o', 's', '^', 'x', 'D', '*', 'v', '>', '<', 'p']

# 2D 可视化
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_lsta[y == label, 0],
                X_lsta[y == label, 1],
                marker=markers[label % len(markers)],
                edgecolor='k',
                facecolor=str(0.5 + 0.05*label),  # 灰度填充
                alpha=0.9,
                label=f'数字 {label}')
plt.xlabel("成分 1")
plt.ylabel("成分 2")
plt.title("LSTA降维结果（2D）")
plt.legend()
plt.show()

# 三维降维
pca_3d = PCA(n_components=3)
X_lsta_3d = pca_3d.fit_transform(X_scaled)

# 3D 可视化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y):
    ax.scatter(X_lsta_3d[y == label, 0],
               X_lsta_3d[y == label, 1],
               X_lsta_3d[y == label, 2],
               marker=markers[label % len(markers)],
               edgecolor='k',
               facecolor=str(0.5 + 0.05*label),  # 灰度填充
               alpha=0.9,
               label=f'数字 {label}')
ax.set_xlabel("成分 1")
ax.set_ylabel("成分 2")
ax.set_zlabel("成分 3")
ax.set_title("LSTA降维结果（3D）")
ax.legend()
plt.show()