import numpy as np
# 总样本数
n_samples = 100

# 统计 TP, TN, FP, FN
TP = 10  # 真正例（预测恶性，实际恶性）
FP = 5   # 假正例（预测恶性，实际良性）
FN = 5   # 假负例（预测良性，实际恶性）
TN = 80  # 真负例（预测良性，实际良性）

# 计算精度 Accuracy（手动实现）
def custom_accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)  # 统计预测正确的样本数量
    total_samples = len(y_true)  # 总样本数
    return correct_predictions / total_samples  # 计算精度

# 生成 y_true 和 y_pred
y_true = np.array([1] * TP + [0] * FP + [1] * FN + [0] * TN)  # 真实标签
y_pred = np.array([1] * (TP + FP) + [0] * (FN + TN))  # 预测标签

# 计算精度
accuracy = custom_accuracy_score(y_true, y_pred)
print(f"手动实现的 Accuracy: {accuracy:.2f}")