import numpy as np
from scipy.stats import rankdata, find_repeats, chi2
from collections import namedtuple

# 定义返回结果的命名元组
FriedmanchisquareResult = namedtuple('FriedmanchisquareResult', ('statistic', 'pvalue'))
def friedmanchisquare(*samples):
    k = len(samples)
    if k < 3:
        raise ValueError('At least 3 sets of samples must be given '
                         f'for Friedman test, got {k}.')
    # 找到最小样本长度
    min_length = min(len(s) for s in samples)
    if min_length < 1:
        raise ValueError('Samples must contain at least one observation.')
    # 截断所有样本到最小长度
    trimmed_samples = [s[:min_length] for s in samples]
    n = min_length  # 数据集数量
    # 将数据转为numpy数组并按行排列（每行是一个数据集）
    data = np.vstack(trimmed_samples).T
    data = data.astype(float)
    # 对每行数据计算秩次
    ranks = np.zeros_like(data)
    for i in range(len(data)):
        ranks[i] = rankdata(data[i])
    # 处理平局（ties）
    ties = 0
    for d in ranks:
        replist, repnum = find_repeats(np.array(d))
        for t in repnum:
            ties += t * (t * t - 1)
    c = 1 - ties / (k * (k * k - 1) * n) if ties > 0 else 1
    # 计算Friedman统计量
    ssbn = np.sum(ranks.sum(axis=0) ** 2)  # 秩次和的平方和
    chisq = (12.0 / (k * n * (k + 1)) * ssbn - 3 * n * (k + 1)) / c
    # 返回统计量和p值
    p_value = chi2.sf(chisq, k - 1)  # 卡方分布的生存函数
    return FriedmanchisquareResult(chisq, p_value), ranks

def nemenyi_test(ranks, n, k, alpha=0.05):
    # q值表（近似值，适用于k<=10时的双尾检验）
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_alpha.get(k, 3.164)  # 默认使用k=10的q值作为近似
    # 计算临界差CD
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    print(f"\nNemenyi检验临界差 (CD) = {cd:.4f} (alpha={alpha})")
    # 计算平均秩次
    avg_ranks = ranks.mean(axis=0)
    print("各算法平均秩次:", [f"{r:.2f}" for r in avg_ranks])
    # 两两比较
    print("\n两两比较结果:")
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            significant = diff > cd
            print(f"算法 {i+1} vs 算法 {j+1}: 秩次差 = {diff:.4f}, {'显著差异' if significant else '无显著差异'}")

# 示例数据：三种算法在五个数据集上的性能
algorithm1 = [0.59, 0.5, 0.57, 0.56, 0.58]
algorithm2 = [0.92, 0.89, 0.88, 0.87, 0.91]
algorithm3 = [0.88, 0.86, 0.89, 0.84, 0.85]

# 执行Friedman检验
result, ranks = friedmanchisquare(algorithm1, algorithm2, algorithm3)
print(f"Friedman统计量: {result.statistic:.4f}")
print(f"p值: {result.pvalue:.4f}")
alpha = 0.05
if result.pvalue < alpha:
    print(f"p值 ({result.pvalue:.4f}) < {alpha}，拒绝零假设，算法之间存在显著差异")
    # 执行Nemenyi后续检验
    nemenyi_test(ranks, len(algorithm1), len([algorithm1, algorithm2, algorithm3]), alpha)
else:
    print(f"p值 ({result.pvalue:.4f}) >= {alpha}，未拒绝零假设，算法之间无显著差异")