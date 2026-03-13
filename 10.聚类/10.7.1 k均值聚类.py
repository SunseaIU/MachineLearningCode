import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os
import sys
import matplotlib.pyplot as plt

def save_iteration(iteration_num, labels, centers, folder='./out/'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    df['label'] = labels
    for label in np.unique(labels):
        cluster_data = df[df['label'] == label].iloc[:, :-1]
        cluster_data.to_csv(os.path.join(folder, f'cluster{label}.csv'), index=False)

    np.savetxt(os.path.join(folder, 'means.csv'), centers, delimiter=',')
    np.savetxt(os.path.join(folder, 'variances.csv'), [np.var(centers, axis=0)], delimiter=',')

if __name__ == "__main__":
    # 默认文件路径和k值
    filename = 'D:/测试/data5000.csv'
    k = 5

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            k = int(sys.argv[2])

    if not os.path.exists(filename):
        print("Error: File not found.")
        sys.exit(-1)

    print('filename:', filename)
    print('k:', k)

    df = pd.read_csv(filename)
    data = df.to_numpy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    print('Finished after %d iterations' % kmeans.n_iter_)

    if os.path.exists('./out/'):
        for f in os.listdir('./out/'):
            os.remove(os.path.join('./out/', f))

    # ---------- 中文显示设置 ----------
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示

    # ---------- 绘图 ----------
    plt.figure(figsize=(10, 8))

    # 定义不同簇的形状符号
    markers = ['o', 's', '^', 'P', 'D', '*', 'X', 'v', '<', '>']  # 可根据需要增加

    # 为每个簇单独绘制散点，颜色和形状都不同
    for i, label in enumerate(np.unique(labels)):
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=f'簇 {label}',
                    marker=markers[i % len(markers)],  # 超过长度则循环使用
                    s=50, alpha=0.7)

    # 聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='聚类中心')

    plt.title('K-Means 聚类')  # 中文标题
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()  # 显示图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()