from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1) 生成同心圆数据
X, y = make_circles(n_samples=400, factor=.3, noise=.05, random_state=0)

# 2) 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 3) 训练 SVM（RBF 核）
svm_clf = SVC(kernel='rbf')
svm_clf.fit(X_train, y_train)

# 4) 测试集预测与准确率
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 5) 创建网格，用于绘制决策边界
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500),
                     np.linspace(-1.5, 1.5, 500))
Z = svm_clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 6) 支持向量及其标签
sv = svm_clf.support_vectors_
sv_labels = y_train[svm_clf.support_]

# 7) 绘图
plt.figure(figsize=(12, 5))

# -------------------------
# (1) 左图：训练数据 + 支持向量
# -------------------------
plt.subplot(1, 2, 1)

# 蓝色类（y=0）：圆形
plt.scatter(X_train[y_train == 0][:, 0],
            X_train[y_train == 0][:, 1],
            c='blue', marker='o', edgecolors='k', s=40, label='类别 0')

# 红色类（y=1）：正方形
plt.scatter(X_train[y_train == 1][:, 0],
            X_train[y_train == 1][:, 1],
            c='red', marker='s', edgecolors='k', s=40, label='类别 1')

# 支持向量：按类别分别画（蓝色空心圆、红色空心方）
plt.scatter(sv[sv_labels == 0][:, 0],
            sv[sv_labels == 0][:, 1],
            s=150, facecolors='none', edgecolors='blue',
            marker='o', linewidths=1.6, label='支持向量 类别 0')

plt.scatter(sv[sv_labels == 1][:, 0],
            sv[sv_labels == 1][:, 1],
            s=150, facecolors='none', edgecolors='red',
            marker='s', linewidths=1.6, label='支持向量 类别 1')

plt.legend()
plt.title('训练数据及支持向量')

# -------------------------
# (2) 右图：决策边界 + 支持向量
# -------------------------
plt.subplot(1, 2, 2)

# 决策边界与间隔线
plt.contour(xx, yy, Z,
            colors='k', levels=[-1, 0, 1],
            alpha=0.6, linestyles=['--', '-', '--'])

# 蓝色类（y=0）
plt.scatter(X_train[y_train == 0][:, 0],
            X_train[y_train == 0][:, 1],
            c='blue', marker='o', edgecolors='k', s=40, label='类别 0')

# 红色类（y=1）
plt.scatter(X_train[y_train == 1][:, 0],
            X_train[y_train == 1][:, 1],
            c='red', marker='s', edgecolors='k', s=40, label='类别 1')

# 支持向量：按类别分别画
plt.scatter(sv[sv_labels == 0][:, 0],
            sv[sv_labels == 0][:, 1],
            s=150, facecolors='none', edgecolors='blue',
            marker='o', linewidths=1.6, label='支持向量 类别 0')

plt.scatter(sv[sv_labels == 1][:, 0],
            sv[sv_labels == 1][:, 1],
            s=150, facecolors='none', edgecolors='red',
            marker='s', linewidths=1.6, label='支持向量 类别 1')

plt.legend()
plt.title('SVM决策边界与支持向量')

plt.tight_layout()
plt.show()