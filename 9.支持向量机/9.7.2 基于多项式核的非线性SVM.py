import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 只取前两个特征
y = iris.target

# 划分训练集和测试集，比例50%:50%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

# 创建基于多项式核的SVM分类器
svm_classifier = SVC(kernel='poly', degree=3)
svm_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = svm_classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# 绘制决策边界
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)

# 用不同形状表示三个类别
markers = ['o', '^', 's']  # 圆形、三角形、方形
labels = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

for i, label in enumerate(labels):
    plt.scatter(X[y == i, 0], X[y == i, 1],
                marker=markers[i], edgecolor='k',
                alpha=0.8, s=80, label=label)

plt.title('多项式核非线性SVM分类')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.legend(title='类别')
plt.show()