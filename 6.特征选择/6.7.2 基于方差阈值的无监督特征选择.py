import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# 加载 Breast Cancer 数据集
cancer = load_breast_cancer()
X = cancer.data  # 特征矩阵
y = cancer.target  # 标签
# 计算每个特征的方差
variances = np.var(X, axis=0) 
# 选择方差大于阈值的特征（设置一个阈值，这里设定阈值为 1）
threshold = 1
selected_features = variances > threshold  # 选择方差大于阈值的特征
# 选择特征
X_selected = X[:, selected_features]  # 选取方差大于阈值的特征
# 打印保留下来的特征的方差
print(f"选择的特征索引: {np.where(selected_features)[0]}")
print(f"每个特征的方差: {variances[selected_features]}")
# 数据集拆分：训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_selected, X_test_selected = X_train[:, selected_features], X_test[:, selected_features]
# 不做特征选择时的分类
clf_all = LogisticRegression(max_iter=10000)
clf_all.fit(X_train, y_train)
y_pred_all = clf_all.predict(X_test)
accuracy_all = accuracy_score(y_test, y_pred_all)
print(f"不做特征选择时的分类准确率: {accuracy_all * 100:.2f}%")
# 做特征选择后的分类
clf_selected = LogisticRegression(max_iter=10000)
clf_selected.fit(X_train_selected, y_train)
y_pred_selected = clf_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"做特征选择后的分类准确率: {accuracy_selected * 100:.2f}%")