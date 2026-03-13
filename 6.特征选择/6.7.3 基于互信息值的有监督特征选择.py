import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示正常
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===== 计算互信息 =====
def mutual_info(x, y, bins=10):
    x_discrete = np.digitize(x, bins=np.histogram_bin_edges(x, bins=bins))
    y_discrete = np.digitize(y, bins=np.histogram_bin_edges(y, bins=bins))
    joint_prob = np.histogram2d(x_discrete, y_discrete, bins=bins)[0]
    joint_prob /= joint_prob.sum()
    x_prob = np.sum(joint_prob, axis=1)
    y_prob = np.sum(joint_prob, axis=0)
    mi = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
    return mi

# ===== 加载数据 =====
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# ===== 特征名英文→中文映射 =====
feature_name_map = {
    "mean radius": "平均半径",
    "mean texture": "平均纹理",
    "mean perimeter": "平均周长",
    "mean area": "平均面积",
    "mean smoothness": "平均光滑度",
    "mean compactness": "平均紧致度",
    "mean concavity": "平均凹度",
    "mean concave points": "平均凹点数",
    "mean symmetry": "平均对称性",
    "mean fractal dimension": "平均分形维数",
    "radius error": "半径误差",
    "texture error": "纹理误差",
    "perimeter error": "周长误差",
    "area error": "面积误差",
    "smoothness error": "光滑度误差",
    "compactness error": "紧致度误差",
    "concavity error": "凹度误差",
    "concave points error": "凹点数误差",
    "symmetry error": "对称性误差",
    "fractal dimension error": "分形维数误差",
    "worst radius": "最大半径",
    "worst texture": "最大纹理",
    "worst perimeter": "最大周长",
    "worst area": "最大面积",
    "worst smoothness": "最大光滑度",
    "worst compactness": "最大紧致度",
    "worst concavity": "最大凹度",
    "worst concave points": "最大凹点数",
    "worst symmetry": "最大对称性",
    "worst fractal dimension": "最大分形维数"
}

feature_names_cn = [feature_name_map[f] for f in feature_names]

# ===== 计算互信息值 =====
mi_scores = [mutual_info(X[:, i], y) for i in range(X.shape[1])]
mi_scores_df = pd.DataFrame({'Feature': feature_names_cn, 'MI Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)

# ===== 选择前 K 个特征并训练逻辑回归 =====
k = 10
selected_features = mi_scores_df.head(k)['Feature'].values
selected_indices = [feature_names_cn.index(feat) for feat in selected_features]
X_selected = X[:, selected_indices]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型在测试集上的准确率: {accuracy:.4f}")

# ===== 可视化：完整特征排序 =====
plt.figure(figsize=(14, 8))
sns.barplot(x="Feature", y="MI Score", data=mi_scores_df, palette="viridis")
plt.xticks(rotation=60, ha="right")  # 旋转60°
plt.xlabel("特征")
plt.ylabel("互信息值")
plt.title("所有特征与目标变量的互信息排序")
plt.tight_layout()
plt.show()

# ===== 可视化：前10个特征 =====
plt.figure(figsize=(10, 6))
sns.barplot(x="Feature", y="MI Score", data=mi_scores_df.head(10), palette="magma")
plt.xticks(rotation=45, ha="right")
plt.xlabel("特征")
plt.ylabel("互信息值")
plt.title("前10个最重要特征（互信息）")
plt.tight_layout()
plt.show()