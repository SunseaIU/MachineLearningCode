import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# 1. 数据准备
# 构造虚拟泰坦尼克号样本数据
np.random.seed(42)
size = 200
sex = np.random.choice([0, 1], size=size)
age = np.random.uniform(18, 60, size=size)
pclass = np.random.choice([1, 2, 3], size=size)
fare = np.random.uniform(10, 100, size=size)
survived = np.random.choice([0, 1], size=size)

df = pd.DataFrame({
    'Sex': sex,
    'Age': age,
    'Pclass': pclass,
    'Fare': fare,
    'Survived': survived
})

# 特征与标签
X = df[['Sex', 'Age', 'Pclass', 'Fare']]
y = df['Survived']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)



# 2. Bagging 实践：随机森林
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)



# 3. Boosting 实践：AdaBoost
ada = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)

# 4. Boosting 实践：GBDT
gbdt = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbdt.fit(X_train, y_train)
gbdt_pred = gbdt.predict(X_test)
gbdt_acc = accuracy_score(y_test, gbdt_pred)


result_df = pd.DataFrame({
    '方法': ['Random Forest', 'AdaBoost', 'GBDT'],
    '测试准确率': [rf_acc, ada_acc, gbdt_acc]
})


# 5. 输出
print("集成学习方法实践案例结果：\n")
print(result_df.to_string(index=False))