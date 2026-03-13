import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 加载糖尿病数据集（此处使用sklearn内置数据集示例，实际需替换为Pima数据集）
X, y = load_diabetes(return_X_y=True)
y = np.where(y > np.median(y), 1, 0)  # 转换为二分类问题
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. L1正则化特征选择
l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
l1_model.fit(X_train, y_train)
l1_coef = l1_model.coef_.ravel()

# 2. L21正则化（组稀疏）实现（示例代码）
class L21Classifier(SGDClassifier):
    def __init__(self, alpha=0.1):
        super().__init__(penalty='elasticnet', loss='log_loss', alpha=alpha, l1_ratio=0.5)

    def fit(self, X, y):
        super().fit(X, y)
        n_features = self.coef_.shape[1]
        self.coef_ = self.coef_.reshape(-1, 2)  # 按2个特征分组
        group_norms = np.linalg.norm(self.coef_, axis=1, keepdims=True)
        self.coef_ = np.where(group_norms < 0.1, 0, self.coef_)
        self.coef_ = self.coef_.reshape(1, n_features)
        return self

# 使用修正后的L21Classifier
l21_model = L21Classifier(alpha=0.1)
l21_model.fit(X_train, y_train)
l21_coef = l21_model.coef_.ravel()

# 可视化特征权重（中文显示）
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.stem(l1_coef)
plt.title("L1正则化特征权重")
plt.xlabel("特征索引")
plt.ylabel("权重")

plt.subplot(1, 2, 2)
plt.stem(l21_coef)
plt.title("L21正则化特征权重")
plt.xlabel("特征索引")
plt.ylabel("权重")

plt.tight_layout()
plt.show()

# 测试集预测
y_pred_l1 = l1_model.predict(X_test)
y_pred_l21 = l21_model.predict(X_test)

print(f"L1正则化分类准确率：{accuracy_score(y_test, y_pred_l1) * 100:.2f}%")
print(f"L21正则化分类准确率：{accuracy_score(y_test, y_pred_l21) * 100:.2f}%")