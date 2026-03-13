import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score


# 求输入样本的类内离散度
def Sw_matrix(sample):
    n, m = sample.shape
    Sw = np.zeros((m, m))
    mu = np.mean(sample, axis=0).reshape(m, 1)
    for i in range(n):
        S = (sample[i, :].reshape(m, 1) - mu) @ (sample[i, :].reshape(m, 1) - mu).T
        Sw = Sw + S
    return Sw


# 求fisher投影方向
def fisher(sample1, sample2):
    n1, m1 = sample1.shape
    n2, m2 = sample2.shape
    mu1 = np.mean(sample1, axis=0).reshape(m1, 1)
    mu2 = np.mean(sample2, axis=0).reshape(m2, 1)
    Sw1 = Sw_matrix(sample1)
    Sw2 = Sw_matrix(sample2)
    Sw = Sw1 + Sw2
    invSw = np.linalg.pinv(Sw)  # 因为Sw可能不可逆，故这里求解其伪逆矩阵；若其可逆，则其逆矩阵和伪逆矩阵相同
    w = invSw @ (mu1 - mu2)
    w = w / np.linalg.norm(w)
    return w


# 投影方向对比可视化
def visualization(sample1, sample2, w, z0):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('分类结果可视化')
    plt.plot(sample1[:, 0], sample1[:, 1], 'r^', sample2[:, 0], sample2[:, 1], 'b*')
    plt.axline((0, 0), slope=(w[1] / w[0])[0], color='y', linestyle='--')
    plt.axline((z0[0][0] * w[0][0], z0[0][0] * w[1][0]), slope=-(w[0] / w[1])[0], color='k', linestyle='-.')
    plt.plot((w.T @ sample1.T * w)[0], (w.T @ sample1.T * w)[1], 'ro', markersize=3)
    plt.plot((w.T @ sample2.T * w)[0], (w.T @ sample2.T * w)[1], 'bs', markersize=3)
    plt.gca().set_aspect(1)
    plt.show()


# 对样本进行预测
def predict(sample, z0, w):
    zz = w.T @ sample.T
    y = np.zeros(len(zz[0]))
    for i in range(len(zz[0])):
        if (zz[0][i] >= z0):
            y[i] = 1
        else:
            y[i] = 2
    return y


def main():
    n1 = 100
    n2 = 100
    m1 = 2
    m2 = 2
    rot = np.array([[1, 0.6], [0.6, 1]])
    sample1 = np.random.randn(n1, m1) @ rot
    sample2 = np.random.randn(n2, m2) @ rot + np.matlib.repmat([0, 2.5], n2, 1)
    mu1 = np.mean(sample1, axis=0).reshape(m1, 1)
    mu2 = np.mean(sample2, axis=0).reshape(m2, 1)

    w = fisher(sample1, sample2)
    z1 = w.T @ mu1
    z2 = w.T @ mu2
    z0 = (z1 + z2) / 2

    # 分类结果显示
    visualization(sample1, sample2, w, z0)

    # 预测
    sample11 = np.random.randn(70, m1) @ rot
    sample22 = np.random.randn(70, m2) @ rot + np.matlib.repmat([0, 2.5], 70, 1)
    true_y11 = np.ones(70)
    true_y22 = np.ones(70) * 2
    true_y = np.append(true_y11, true_y22)
    pre_y = predict(np.vstack((sample11, sample22)), z0, w)
print("The classification accuracy of FLD is ", round(accuracy_score(true_y, pre_y) * 100, 2), "%")

if __name__ == '__main__':
    main()