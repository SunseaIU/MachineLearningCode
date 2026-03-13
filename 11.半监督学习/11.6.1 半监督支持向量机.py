import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class S3VM:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', max_iter=100):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

    def fit(self, X, y):
        """
        训练半监督S3VM模型
        :param X: 特征矩阵
        :param y: 标签矩阵（包含有标签数据和无标签数据）
        """
        prev_y = np.copy(y)  # 保存上次迭代的标签

        for i in range(self.max_iter):
            # 训练 SVM 模型（只使用有标签数据）
            labeled_X = X[y != -1]
            labeled_y = y[y != -1]
            self.model.fit(labeled_X, labeled_y)

            # 用当前模型对无标签数据进行预测
            unlabeled_X = X[y == -1]

            if len(unlabeled_X) == 0:
                print("No unlabeled data left to predict!")
                break

            predicted_y = self.model.predict(unlabeled_X)

            # 更新无标签数据的标签
            y[y == -1] = predicted_y

            # 检查是否更新了标签
            if np.array_equal(prev_y, y):
                print(f"Iteration {i + 1}: No label change. Converged.")
                break

            prev_y = np.copy(y)

            # 输出迭代过程中的一些信息（可选）
            accuracy = accuracy_score(labeled_y, self.model.predict(labeled_X))
            print(f"Iteration {i + 1}: Accuracy on labeled data = {accuracy:.2f}")

    def predict(self, X):
        """
        预测给定数据的标签
        :param X: 特征矩阵
        :return: 预测的标签
        """
        return self.model.predict(X)

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集分为有标签数据和无标签数据
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.9, random_state=42)

# 将无标签数据的标签设置为 -1
y_unlabeled = -1 * np.ones(len(y_unlabeled))

# 合并有标签和无标签数据
X_train = np.vstack([X_labeled, X_unlabeled])
y_train = np.hstack([y_labeled, y_unlabeled])

# 初始化 S3VM 模型
s3vm = S3VM(kernel='linear', C=1.0, gamma='scale', max_iter=200)  # 增加最大迭代次数

# 训练 S3VM 模型
s3vm.fit(X_train, y_train)

# 获取预测标签
predicted_labels = s3vm.predict(X_unlabeled)

# 评估模型性能
y_pred_labeled = s3vm.predict(X_labeled)
accuracy = accuracy_score(y_labeled, y_pred_labeled)
print(f"Accuracy on labeled data: {accuracy:.2f}")

# 输出无标签数据的预测标签
print(f"Predicted labels for unlabeled data: {predicted_labels}")

# 可视化（使用前两个特征）
X_2d = X[:, :2]
X_labeled_2d = X_labeled[:, :2]
X_unlabeled_2d = X_unlabeled[:, :2]

s3vm_2d = S3VM(kernel='linear', C=1.0, gamma='scale', max_iter=200)
s3vm_2d.fit(np.vstack([X_labeled_2d, X_unlabeled_2d]), np.hstack([y_labeled, y_unlabeled]))

# 创建网格
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点
Z = s3vm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)