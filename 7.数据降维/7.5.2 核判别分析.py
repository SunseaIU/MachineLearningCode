import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import seaborn as sns
from typing import Tuple, Dict, Optional


class KernelDiscriminantAnalysis:
    """核判别分析实现类"""

    def __init__(self, n_components: int = 1, kernel: str = 'rbf',
                 kernel_params: Optional[Dict] = None, reg_param: float = 1e-6):
        """
        初始化KDA

        参数:
        n_components: 降维后的维度
        kernel: 核函数类型 ('rbf', 'poly', 'linear')
        kernel_params: 核函数参数
        reg_param: 正则化参数
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.reg_param = reg_param
        self.fitted = False

    def _get_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        if self.kernel == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0)
            dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                   np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * dist)
        elif self.kernel == 'poly':
            degree = self.kernel_params.get('degree', 3)
            return (1 + np.dot(X1, X2.T)) ** degree
        elif self.kernel == 'linear':
            return np.dot(X1, X2.T)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelDiscriminantAnalysis':
        """
        训练KDA模型

        参数:
        X: 训练数据
        y: 标签
        """
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

        # 计算核矩阵
        self.K = self._get_kernel(X, X)

        # 中心化核矩阵
        N = self.K.shape[0]
        one_n = np.ones((N, N)) / N
        self.K_centered = self.K - one_n @ self.K - \
                          self.K @ one_n + one_n @ self.K @ one_n

        # 计算类内和类间散度矩阵
        self._compute_scatter_matrices()

        # 求解广义特征值问题
        try:
            eigvals, eigvecs = eigh(self.SB, self.SW)
            # 选择最大的n_components个特征值
            idx = np.argsort(eigvals)[::-1][:self.n_components]
            self.eigvals = eigvals[idx]
            self.eigvecs = eigvecs[:, idx]
            self.fitted = True
        except np.linalg.LinAlgError as e:
            print(f"Error in eigenvalue decomposition: {e}")
            self.fitted = False

        return self

    def _compute_scatter_matrices(self):
        """计算散度矩阵"""
        N = self.K_centered.shape[0]
        self.SB = np.zeros((N, N))
        self.SW = np.zeros((N, N))

        for c in self.classes:
            mask = self.y_train == c
            K_c = self.K_centered[mask]
            n_c = K_c.shape[0]

            # 计算类内散度
            K_c_centered = K_c - K_c.mean(axis=0)
            self.SW += K_c_centered.T @ K_c_centered

            # 计算类间散度
            m_c = self.K_centered[mask].mean(axis=0)
            m = self.K_centered.mean(axis=0)
            self.SB += n_c * (m_c - m).reshape(-1, 1) @ (m_c - m).reshape(1, -1)

            # 添加正则化
        self.SW += self.reg_param * np.eye(N)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """将数据投影到KDA空间"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")

        K_test = self._get_kernel(X, self.X_train)
        return K_test @ self.eigvecs

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)


class KDAVisualizer:
    """KDA可视化类（适合黑白印刷版）"""

    @staticmethod
    def plot_results(kda: KernelDiscriminantAnalysis, X: np.ndarray,
                     y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """绘制结果（黑白印刷友好版）"""
        plt.figure(figsize=(15, 5))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 定义符号列表，不同类别不同符号
        markers = ['o', 's', '^', 'x', 'D']  # 可扩展更多类别

        # ---------- 原始数据 ----------
        plt.subplot(131)
        for idx, cls in enumerate(np.unique(y)):
            plt.scatter(X[y == cls, 0], X[y == cls, 1],
                        marker=markers[idx % len(markers)],
                        edgecolor='k', facecolor='white',  # 黑白印刷适配
                        label=f'类别 {cls}')
        plt.title("原始数据分布")
        plt.xlabel("特征1")
        plt.ylabel("特征2")
        plt.legend()

        # ---------- KDA投影结果 ----------
        plt.subplot(132)
        X_transformed = kda.transform(X)
        if X_transformed.shape[1] >= 2:
            for idx, cls in enumerate(np.unique(y)):
                plt.scatter(X_transformed[y == cls, 0],
                            X_transformed[y == cls, 1],
                            marker=markers[idx % len(markers)],
                            edgecolor='k', facecolor='white',
                            label=f'类别 {cls}')
        else:
            for idx, cls in enumerate(np.unique(y)):
                plt.scatter(X_transformed[y == cls],
                            np.zeros_like(X_transformed[y == cls]),
                            marker=markers[idx % len(markers)],
                            edgecolor='k', facecolor='white',
                            label=f'类别 {cls}')
        plt.title("KDA投影结果")
        plt.xlabel("KDA成分1")
        plt.ylabel("KDA成分2" if X_transformed.shape[1] >= 2 else "")
        plt.legend()

        # ---------- 决策边界 ----------
        plt.subplot(133)
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5,
                                         X[:, 0].max() + 0.5, 200),
                             np.linspace(X[:, 1].min() - 0.5,
                                         X[:, 1].max() + 0.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = kda.transform(grid)

        # 使用第一主成分作为判别标准绘制灰度决策边界
        if Z.ndim > 1:
            Z_plot = Z[:, 0].reshape(xx.shape)
        else:
            Z_plot = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z_plot, levels=10, cmap='Greys', alpha=0.3)  # 灰度填充

        for idx, cls in enumerate(np.unique(y_test)):
            plt.scatter(X_test[y_test == cls, 0],
                        X_test[y_test == cls, 1],
                        marker=markers[idx % len(markers)],
                        edgecolor='k', facecolor='white',
                        label=f'类别 {cls}')
        plt.title("决策边界")
        plt.xlabel("特征1")
        plt.ylabel("特征2")
        plt.legend()

        plt.tight_layout()
        plt.show()



def main():
    # 生成数据
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # 改进数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建并训练KDA模型 - 修改核函数参数
    kda = KernelDiscriminantAnalysis(
        n_components=1,
        kernel='rbf',  # 使用RBF核
        kernel_params={'gamma': 0.5},  # 降低gamma值，避免过拟合
        reg_param=1e-4  # 增加正则化参数
    )
    kda.fit(X_train_scaled, y_train)

    # 预测 - 改进阈值选择方法
    X_train_transformed = kda.transform(X_train_scaled)
    X_test_transformed = kda.transform(X_test_scaled)

    # 使用更好的阈值选择方法
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_train_transformed)
    # 使用GMM进行预测
    y_train_pred = gmm.predict(X_train_transformed)
    y_test_pred = gmm.predict(X_test_transformed)
    # 如果需要，翻转标签以匹配原始标签
    if accuracy_score(y_train, y_train_pred) < 0.5:
        y_train_pred = 1 - y_train_pred
        y_test_pred = 1 - y_test_pred
        # 计算性能指标
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # 计算更多评估指标
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average='binary')
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='binary')
    # 可视化结果
    KDAVisualizer.plot_results(kda, X_train_scaled, y_train,
                               X_test_scaled, y_test)
    # 打印详细结果
    print("\n训练集评估指标:")
    print(f"准确率: {train_acc * 100:.2f}%")
    print(f"精确率: {train_precision * 100:.2f}%")
    print(f"召回率: {train_recall * 100:.2f}%")
    print(f"F1分数: {train_f1 * 100:.2f}%")
    print("\n测试集评估指标:")
    print(f"准确率: {test_acc * 100:.2f}%")
    print(f"精确率: {test_precision * 100:.2f}%")
    print(f"召回率: {test_recall * 100:.2f}%")
    print(f"F1分数: {test_f1 * 100:.2f}%")
if __name__ == "__main__":
    main()