import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 生成模拟数据（两个高斯分布）
np.random.seed(0)
n_samples = 300
mu1 = np.array([0, 0])
mu2 = np.array([5, 5])
cov1 = np.array([[1, 0.5], [0.5, 1]])
cov2 = np.array([[1, -0.3], [-0.3, 1]])

X1 = np.random.multivariate_normal(mu1, cov1, n_samples)
X2 = np.random.multivariate_normal(mu2, cov2, n_samples)
X = np.vstack((X1, X2))  # 合并数据，共 600 个点
n, d = X.shape

# 假设有两个高斯分布
k = 2

# 初始化参数
np.random.seed(42)
mu = X[np.random.choice(n, k, replace=False)]         # 初始化均值
cov = [np.eye(d) for _ in range(k)]                  # 初始化协方差矩阵
pi = np.ones(k) / k                                  # 初始化混合系数（均等）
gamma = np.zeros((n, k))                             # 软分配矩阵

# EM 算法主循环
max_iters = 100
tol = 1e-4
log_likelihoods = []
for iteration in range(max_iters):
    # -------- E步 --------
    for i in range(k):
        gamma[:, i] = pi[i] * multivariate_normal.pdf(X, mean=mu[i], cov=cov[i])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # -------- M步 --------
    N_k = gamma.sum(axis=0)
    for i in range(k):
        mu[i] = (gamma[:, i][:, np.newaxis] * X).sum(axis=0) / N_k[i]
        x_mu = X - mu[i]
        cov[i] = np.dot((gamma[:, i][:, np.newaxis] * x_mu).T, x_mu) / N_k[i]
        pi[i] = N_k[i] / n
    # -------- Log-likelihood --------
    log_likelihood = np.sum(np.log(np.sum([
        pi[j] * multivariate_normal.pdf(X, mean=mu[j], cov=cov[j])
        for j in range(k)
    ], axis=0)))
    log_likelihoods.append(log_likelihood)

    if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
        print(f'EM converged at iteration {iteration}')
        break
# 结果可视化
labels = np.argmax(gamma, axis=1)
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], c='blue', marker='o', alpha=0.6, label='Cluster 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c='red', marker='s', alpha=0.6, label='Cluster 2')
plt.scatter([m[0] for m in mu], [m[1] for m in mu], c='black', marker='x', s=100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.grid(True)
plt.show()