import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==========================
# 中文显示设置（解决乱码）
# ==========================
rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False    # 解决负号显示

# ==========================
# 数据处理函数
# ==========================
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def center(X):
    return X - np.mean(X, axis=0)

def pca(X, n_components, use_standardization=True):
    X_preprocessed = standardize(X) if use_standardization else center(X)
    cov_matrix = np.cov(X_preprocessed, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]
    components = eigenvectors[:, :n_components]
    X_pca = X_preprocessed @ components
    return X_pca, eigenvalues[:n_components], components

def whitening_pca(X, n_components, epsilon=1e-6):
    X_pca, eigenvalues, components = pca(X, n_components, use_standardization=False)
    scaling_matrix = np.diag(1.0 / np.sqrt(eigenvalues + epsilon))
    X_whitened = X_pca @ scaling_matrix
    whitening_matrix = components @ scaling_matrix
    return X_whitened, whitening_matrix

# ==========================
# 示例数据
# ==========================
raw_data = np.array([
    [746395.1, 30105.17, 315806.2, 104967.17],
    [832035.9, 31399.49, 347329.7, 124789.81],
    [919281.1, 30727.12, 377783.1, 140880.32],
    [986515.2, 31079.24, 408017.2, 143253.69],
    [1013567.0, 32165.22, 391980.6, 142936.40]
])

X_normalized = standardize(raw_data)
X_pca, var_pca, _ = pca(X_normalized, n_components=2)
X_white, _ = whitening_pca(X_normalized, n_components=2)

# ==========================
# 输出协方差矩阵与方差
# ==========================
def print_covariance(title, data):
    cov = np.cov(data.T)
    print(f"\n{title}协方差矩阵:")
    print(np.round(cov, 3))

print_covariance("原始数据", X_normalized)
print_covariance("普通PCA", X_pca)
print_covariance("白化变换后", X_white)
print("\n普通PCA方差:", np.round(np.var(X_pca, axis=0), 3))
print("白化变换后方差:", np.round(np.var(X_white, axis=0), 3))

# ==========================
# 可视化（彩色）
# ==========================
plt.figure(figsize=(12, 5))

# 普通PCA
plt.subplot(121)
plt.scatter(X_pca[:,0], X_pca[:,1], c='blue', s=60, alpha=0.7)  # 蓝色散点
plt.title("标准PCA\n(方差: {:.3f}, {:.3f})".format(*var_pca))
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.grid(True)

# 白化变换
plt.subplot(122)
plt.scatter(X_white[:,0], X_white[:,1], c='red', s=60, alpha=0.7)  # 红色散点
plt.title("白化变换\n(单位方差)")
plt.xlabel("主成分1（白化变换）")
plt.ylabel("主成分2（白化变换）")
plt.grid(True)

plt.tight_layout()
plt.show()