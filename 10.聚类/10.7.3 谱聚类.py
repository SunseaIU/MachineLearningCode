import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def kernel(x1, x2, sigma):
    return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * sigma ** 2))


def affinity_matrix(X, sigma):
    A = np.zeros((len(X), len(X)))
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            A[i, j] = A[j, i] = kernel(X[i], X[j], sigma)
    return A


def getD(A):
    D = np.zeros(A.shape)
    for i in range(A.shape[0]):
        D[i, i] = np.sum(A[i, :])
    return D

def getL(D, A):
    L = D - A
    return L

def get_eigen(L, num_clusters):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    best_eigenvalues = np.argsort(eigenvalues)[0:num_clusters]
    U = eigenvectors[:, best_eigenvalues]
    return U

def cluster(data, num_clusters, sigma):
    data = np.array(data)
    W = affinity_matrix(data, sigma)
    D = getD(W)
    L = getL(D, W)
    eigenvectors = get_eigen(L, num_clusters)
    clf = KMeans(n_clusters=num_clusters)
    s = clf.fit(eigenvectors)
    label = s.labels_
    return label


# --- 可视化部分 (去除了图例) ---

def plotRes(data, clusterResult, clusterNum):
    n = len(data)
    scatterColors = ['black', 'blue', 'red', 'yellow', 'green', 'purple', 'orange']
    # 定义形状列表：圆圈、方块、三角、菱形、叉号...
    scatterMarkers = ['o', 's', '^', 'D', 'x', '*', 'v']

    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        marker = scatterMarkers[i % len(scatterMarkers)]
        x1 = []
        y1 = []
        for j in range(n):
            if clusterResult[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])

        # 画散点图，不加 label 参数，也不调用 plt.legend()
        plt.scatter(x1, y1, c=color, marker=marker, s=30)


if __name__ == '__main__':
    plt.figure(figsize=(15, 5))
    n_samples = 400  # 样本数


    # 1. 月牙形
    plt.subplot(131)
    cluster_num = 2
    data, target = make_moons(n_samples=n_samples, noise=0.1)
    label = cluster(data, cluster_num, sigma=0.1)
    plotRes(data, label, cluster_num)


    # 2. 圆形
    plt.subplot(132)
    cluster_num = 2
    data, target = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
    label = cluster(data, cluster_num, sigma=0.1)
    plotRes(data, label, cluster_num)


    # 3. 正态分布 (Blobs)
    plt.subplot(133)
    cluster_num = 4
    data, target = make_blobs(n_samples=n_samples, centers=cluster_num, random_state=24)
    # 这里的 sigma 设为 2.0 以保证正确分类
    label = cluster(data, cluster_num, sigma=2.0)
    plotRes(data, label, cluster_num)


    plt.show()