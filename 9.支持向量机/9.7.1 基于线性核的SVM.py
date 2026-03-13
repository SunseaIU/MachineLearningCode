import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成线性可分的数据集
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性核SVM模型
model = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', C=1))
model.fit(X_train, y_train)

# 在测试集上预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率 (ACC): {accuracy:.2f}")

# 绘制决策边界，类别用不同形状
def plot_decision_boundary_shapes(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测网格点类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)

    # 根据类别使用不同形状绘制散点
    markers = ['o', '^']  # 类别0用圆形，类别1用三角形
    for class_value in np.unique(y):
        plt.scatter(X[y == class_value, 0], X[y == class_value, 1],
                    marker=markers[class_value],
                    edgecolor='k', s=80, label=f'类别 {class_value}')

    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('线性核SVM的决策边界')
    plt.legend(title='类别')
    plt.show()

# 调用绘图函数
plot_decision_boundary_shapes(model, X, y)