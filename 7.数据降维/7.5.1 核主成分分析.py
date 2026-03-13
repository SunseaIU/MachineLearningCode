import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        except:
            print("无法设置中文字体，请手动安装相应字体")

# 生成随机数据
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KPCA降维
kpca = KernelPCA(kernel='rbf', gamma=1.0, n_components=1)
X_kpca = kpca.fit_transform(X_scaled).flatten()

# 创建图形
plt.figure(figsize=(12, 6))

# 原始数据可视化（黑白风格，使用不同符号区分类别）
plt.subplot(1, 2, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], marker='o', edgecolor='k', facecolor='none', s=50, label='类别0')
plt.scatter(X[y==1, 0], X[y==1, 1], marker='s', edgecolor='k', facecolor='grey', s=50, label='类别1')
plt.title("原始数据", fontsize=12)
plt.xlabel("特征1", fontsize=10)
plt.ylabel("特征2", fontsize=10)
plt.legend()  # 添加图例说明不同点的含义

# 降维后数据可视化
plt.subplot(1, 2, 2)
plt.scatter(X_kpca[y==0], np.zeros_like(X_kpca[y==0]), marker='o', edgecolor='k', facecolor='none', s=50, label='类别0')
plt.scatter(X_kpca[y==1], np.zeros_like(X_kpca[y==1]), marker='s', edgecolor='k', facecolor='grey', s=50, label='类别1')
plt.title("KPCA降维后的数据", fontsize=12)
plt.xlabel("主成分1", fontsize=10)
plt.yticks([])  # 移除y轴刻度
plt.legend()  # 添加图例说明不同点的含义

# 调整布局并显示
plt.tight_layout()
plt.show()