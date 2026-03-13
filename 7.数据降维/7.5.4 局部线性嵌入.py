import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体，防止报错
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 加载手写数字数据集
digits = datasets.load_digits()
X, y = digits.data, digits.target  # X 是特征，y 是类别

# 可视化原始数据（部分样本）
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"数字 {y[i]}")
    ax.axis('off')
plt.show()

# LLE降维到2维
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)

# 定义不同数字的符号，适合黑白打印
markers = ['o', 's', '^', 'x', 'D', '*', 'v', '>', '<', 'p']

# 2D 可视化
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_lle[y == label, 0],
                X_lle[y == label, 1],
                marker=markers[label % len(markers)],
                edgecolor='k',
                facecolor=str(0.5 + 0.05*label),  # 灰度填充
                alpha=0.9,
                label=f'数字 {label}')
plt.xlabel("成分 1")
plt.ylabel("成分 2")
plt.title("LLE降维结果（2D）")
plt.legend()
plt.show()

# LLE降维到3维
lle_3d = LocallyLinearEmbedding(n_neighbors=10, n_components=3)
X_lle_3d = lle_3d.fit_transform(X)

# 3D 可视化
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y):
    ax.scatter(X_lle_3d[y == label, 0],
               X_lle_3d[y == label, 1],
               X_lle_3d[y == label, 2],
               marker=markers[label % len(markers)],
               edgecolor='k',
               facecolor=str(0.5 + 0.05*label),  # 灰度填充
               alpha=0.9,
               label=f'数字 {label}')
ax.set_xlabel("成分 1")
ax.set_ylabel("成分 2")
ax.set_zlabel("成分 3")
ax.set_title("LLE降维结果（3D）")
ax.legend()
plt.show()