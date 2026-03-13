import numpy as np
import matplotlib.pyplot as plt

def pca(X, num_components):
    # 1. 去中心化处理
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X.T)

    # 3. 计算特征根和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 特征根排序
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # 5. 选择前n个最大的特征根
    eigenvectors = eigenvectors[:,:num_components]

    # 6. 转化到新空间
    X_pca = X @ eigenvectors

    return X_pca, eigenvalues

data = np.array([[746395.1, 30105.17, 315806.2, 104967.17],
                 [832035.9, 31399.49, 347329.7, 124789.81],
                 [949281.1, 30727.12, 377783.1, 140880.32],
                 [986515.2, 31079.24, 408017.2, 143253.69],
                 [1013567.0, 32165.22, 391980.6, 142936.40]])
data_pca, data_eigVal = pca(data, 2)
x = np.arange(1, 5)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.plot(x, data_eigVal, marker='o',markersize=5)
plt.xticks(x)
plt.xlabel("因子个数")
plt.ylabel("特征根")
plt.show()