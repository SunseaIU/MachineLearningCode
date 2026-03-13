import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False    # 解决负号显示

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将类别名改为中文
target_names = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

# LDA降维
fld = LinearDiscriminantAnalysis(n_components=2)
X_r2 = fld.fit(X, y).transform(X)

# 数据点颜色和标记
colors = ['navy', 'turquoise', 'darkorange']
markers = ['o', '^', '*']

# 绘制散点图
plt.figure(figsize=(8,6))
for color, i, target_name, marker in zip(colors, [0, 1, 2], target_names, markers):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color,
                label=target_name, marker=marker)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('鸢尾花数据集的FLD降维结果', fontsize=14)
plt.xlabel('判别主成分1', fontsize=12)
plt.ylabel('判别主成分2', fontsize=12)
plt.grid(True)
plt.show()