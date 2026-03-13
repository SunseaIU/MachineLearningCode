import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class ConstrainedSeedKMeans:
    def __init__(self,
                 n_clusters=2,
                 *,
                 n_init=10,
                 max_iter=300,
                 tol=0.0001,
                 verbose=False,
                 invalide_label=-1):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.INVALID_LABEL = invalide_label
        # n_clusters簇数量 init迭代轮次 max_iter 最大迭代轮次 tol终止条件
        # invalide_label无类别标签 verbose是否打印调试信息

    def _check_params(self, X, y):  # 参数检查
        if type(X) not in (np.ndarray, torch.Tensor):
            raise TypeError(f"Type of X can only take numpy.ndarray and "
                            f"torch.Tensor, but got {type(X)} instead.")

        if type(y) not in (list, np.ndarray, torch.Tensor):
            raise TypeError(f"Type of y can only take list, numpy.ndarray, and"
                            f"torch.Tensor, but got{type(y)} instead.")

        if self.n_clusters > X.shape[0]:
            raise ValueError(f"The number of clusters mube be less than the "
                             f"number of samples.")

        if self.max_iter <= 0:
            raise ValueError(f"The number of maximum iteration must larger than zero.")

    def _init_centroids(self, X, y):  # 初始化聚类中心
        if type(y) == np.ndarray:
            pkg = np
        elif type(y) == torch.Tensor:
            pkg = torch
        elif type(y) == list and type(X) == np.ndarray:
            y = np.array(y)
            pkg = np
        elif type(y) == list and type(X) == torch.Tensor:
            y = torch.Tensor(y)
            pkg = torch
        else:
            raise TypeError('Data type is not supported, please check it again.')

        y_unique = pkg.unique(y)  # 计算已有标签类别数 n_seed_centroids
        if self.INVALID_LABEL in y_unique:
            n_seed_centroids = len(y_unique) - 1
        else:
            n_seed_centroids = len(y_unique)
        assert n_seed_centroids <= self.n_clusters, f"The number of seed centroids" \
                                                    f"should be less than the total" \
                                                    f"number of clusters."

        centers = pkg.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        # 已有标签数据初始化部分聚类中心
        for i in range(n_seed_centroids):
            seed_samples = X[y == i]
            centers[i] = seed_samples.mean(axis=0)

        # 选择未标注样本初始化剩余的聚类中心
        unlabel_idxes = pkg.where(y == self.INVALID_LABEL)[0]  # np.where returns a tuple

        if len(unlabel_idxes) == 0:
            raise ValueError("All samples are labeled! No need for clustering!")

        if len(unlabel_idxes) < self.n_clusters - n_seed_centroids:
            idx = np.random.randint(X.shape[0], size=self.n_clusters - n_seed_centroids)
            print('index', idx)

            for i in range(n_seed_centroids, self.n_clusters):
                centers[i] = X[idx[i - n_seed_centroids]]

        else:
            for i in range(n_seed_centroids, self.n_clusters):
                idx = np.random.choice(unlabel_idxes, 1, replace=False)
                centers[i] = X[idx]

        return centers, n_seed_centroids

    def _kmeans(self, X, y, init_centers):
        indices = copy.copy(y)
        if type(indices) == list:
            indices = np.array(indices)
        n_samples, n_features = X.shape[0], X.shape[1]
        cur_centers = init_centers
        new_centers = copy.deepcopy(init_centers)

        # Main loop
        for iter_ in range(self.max_iter):
            for i in range(n_samples):
                if y[i] != self.INVALID_LABEL:
                    continue

                if type(X) == np.ndarray:
                    min_idx = np.linalg.norm(cur_centers - X[i], axis=1).argmin()
                else:
                    min_idx = torch.norm(cur_centers - X[i], dim=1).argmin()
                indices[i] = min_idx

            # 更新所有聚类中心
            for i in range(self.n_clusters):
                cluster_samples = X[indices == i]
                if cluster_samples.shape[0] == 0:
                    new_centers[i] = X[np.random.choice(n_samples, 1, replace=False)]
                else:
                    new_centers[i] = cluster_samples.mean(axis=0)

            # 计算代价
            inertia = 0
            for i in range(self.n_clusters):
                if type(X) == np.ndarray:
                    inertia += np.linalg.norm(X[indices == i] - new_centers[i], axis=1).sum()
                else:
                    inertia += torch.norm(X[indices == i] - new_centers[i], dim=1).sum().item()
            if self.verbose:
                print('Iteration {}, inertia: {}'.format(iter_, inertia))

            # 检查终止条件
            if type(X) == np.ndarray:
                difference = np.linalg.norm(new_centers - cur_centers, ord='fro')
            else:
                difference = torch.norm(new_centers - cur_centers, p='fro')
            if difference < self.tol:
                if self.verbose:
                    print('Converged at iteration {}.\n'.format(iter_))
                break
            cur_centers = copy.deepcopy(new_centers)

        return new_centers, indices, inertia

    def fit(self, X, y):
        """Using features and little labels to do clustering."""
        self._check_params(X, y)

        _, n_seed_centroids = self._init_centroids(X, y)
        if n_seed_centroids == self.n_clusters:
            self.n_init = 1

        # run constrained seed KMeans n_init times in order to choose the best one
        best_inertia = None
        best_centers, best_indices = None, None
        for i in range(self.n_init):
            init_centers, _ = self._init_centroids(X, y)
            if self.verbose:
                print('Initialization complete')
            new_centers, indices, new_inertia = self._kmeans(X, y, init_centers)
            if best_inertia is None or new_inertia < best_inertia:
                best_inertia = new_inertia
                best_centers = new_centers
                best_indices = indices

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.indices = best_indices

        return self

    def predict(self, X):
        """Predict the associated cluster index of samples."""
        n_samples = X.shape[0]
        indices = [-1 for _ in range(n_samples)]

        for i in range(n_samples):
            if type(X) == np.ndarray:
                min_idx = np.linalg.norm(self.cluster_centers_ - X[i], axis=1).argmin()
            else:
                min_idx = torch.norm(self.cluster_centers_ - X[i], dim=1).argmin()
            indices[i] = min_idx

        if type(X) == np.ndarray:
            return np.array(indices)
        else:
            return torch.tensor(indices)

    def fit_predict(self, X, y):
        """Convenient function."""
        return self.fit(X, y).predict(X)

    def transform(self, X):
        """Transform the input to the centorid space."""
        if type(X) == np.ndarray:
            pkg = np
        else:
            pkg = torch

        n_samples = X.shape[0]
        output = pkg.empty((n_samples, self.n_clusters), dtype=X.dtype)
        for i in range(n_samples):
            if type(X) == np.ndarray:
                output[i] = np.linalg.norm(self.cluster_centers_ - X[i], axis=1)
            else:
                output[i] = torch.norm(self.cluster_centers_ - X[i], dim=1)

        return output

    def fit_transform(self, X, y):
        """Convenient function"""
        return self.fit(X, y).transform(X)

    def score(self, X):
        """Opposite of the value of X on the K-means objective."""
        interia = 0
        n_samples = X.shape[0]

        for i in range(n_samples):
            if type(X) == np.ndarray:
                interia += np.linalg.norm(self.cluster_centers_ - X[i], axis=1).min()
            else:
                interia += torch.norm(self.cluster_centers_ - X[i], dim=1).min().item()

        return -1 * interia


def plot(X, estimator, name):
    df = pd.DataFrame()
    df['dim1'] = X[:, 0]
    df['dim2'] = X[:, 1]

    # 修改：将列名从 'y' 改为 'Cluster'，避免图例标题显示为 'y'
    # 如果想让图例标题完全空白，我们在下面会设置 title=None
    label_col_name = 'Cluster'

    if name == 'sklearn_kmeans':
        df[label_col_name] = estimator.labels_
    else:
        df[label_col_name] = estimator.indices

    # 将标签转换为分类变量
    df[label_col_name] = df[label_col_name].astype('category')

    plt.close()
    plt.xlim(0.1, 0.9)
    plt.ylim(0.0, 0.8)

    # 修改：hue 和 style 使用新的列名
    ax = sns.scatterplot(x='dim1',
                         y='dim2',
                         hue=label_col_name,
                         style=label_col_name,
                         markers=True,
                         palette=sns.color_palette('hls', 3),
                         data=df)

    # 修改：将图例标题设为空，确保不出现 'y' 或其他标题
    ax.legend(title=None)

    plt.show()


if __name__ == '__main__':
    # Load watermelon-4.0 dataset from book Machine Learning by Zhihua Zhou
    try:
        dataset = np.genfromtxt('./watermelon_4.0.txt', delimiter=',')
        X = dataset[:, 1:]  # the first column are IDs
    except OSError:
        print("未找到 watermelon_4.0.txt，生成随机数据用于演示...")
        X = np.random.rand(30, 2) * 0.8 + 0.1  # 生成一些0.1-0.9之间的随机数

    y = [-1 for _ in range(X.shape[0])]  # by default, all samples has no label
    seed_kmeans = ConstrainedSeedKMeans(n_clusters=3, n_init=10, verbose=False)

    # 简单的防止越界检查，仅用于随机数据情况
    if X.shape[0] > 25:
        y[3], y[24] = 0, 0
        y[11], y[19] = 1, 1
        y[13], y[16] = 2, 2

    seed_kmeans.fit(X, y)
    plot(X, seed_kmeans, name='seed_kmeans')