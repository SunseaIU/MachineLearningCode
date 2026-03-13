import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    boston_data, boston_target, test_size=0.2, random_state=42)

# 数据标准化
scaler_x = StandardScaler()
X_train_std = scaler_x.fit_transform(X_train)
X_test_std = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_std = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
y_test_std = scaler_y.transform(y_test.reshape(-1,1)).ravel()

# 设置待测试的参数
param_grid = {"C": [1e0, 1e1, 1e2, 1e3],
              "gamma": np.logspace(-2, 2, 5)}

# 利用GridSearchCV寻找最优参数
model = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid=param_grid)
model.fit(X_train_std, y_train_std)

# 打印最优参数
print("最优参数为 %s，对应的得分为 %0.2f" % (model.best_params_, model.best_score_))

# 做预测
y_pred = model.predict(X_test_std)

# 打印R2分数和均方误差
print('R2分数: ', r2_score(y_test_std, y_pred))
print('均方误差: ', mean_squared_error(y_test_std, y_pred))

# 绘图
plt.scatter(y_test_std, y_pred, color='red')
plt.plot([y_test_std.min(), y_test_std.max()],
         [y_test_std.min(), y_test_std.max()], 'k--', lw=3)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('SVR预测结果')
plt.grid()
plt.show()