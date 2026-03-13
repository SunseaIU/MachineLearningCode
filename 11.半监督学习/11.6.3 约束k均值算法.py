import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calDistance(centers, nodes):
    '''
    计算节点间距离
    输入：centers-中心，nodes-节点；
    输出：距离矩阵-dis_matrix
    '''
    dis_matrix = pd.DataFrame(data=None, columns=range(len(centers)), index=range(len(nodes)))
    for i in range(len(nodes)):
        xi, yi = nodes[i][0], nodes[i][1]
        for j in range(len(centers)):
            xj, yj = centers[j][0], centers[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return dis_matrix


def scatter_diagram(clusters, nodes):
    '''
    #画路径图
    输入：nodes-节点坐标；
    输出：散点图
    '''
    # 定义形状列表：圆、方、上三角、菱形、下三角、叉、星、五边形
    markers = ['o', 's', '^', 'D', 'v', 'x', '*', 'p']

    # enumerate用于同时获取索引i和数据cluster
    for i, cluster in enumerate(clusters):
        x, y = [], []
        for Coordinate in cluster:
            x.append(Coordinate[0])
            y.append(Coordinate[1])

        # 使用取余运算防止簇的数量超过形状列表的长度
        marker_style = markers[i % len(markers)]

        # 绘制散点，指定marker
        plt.scatter(x, y, alpha=0.8, marker=marker_style, label=f'簇 {i + 1}')

    plt.xlabel('x')
    plt.ylabel('y')
    # 如果想看图例可以取消下面这行的注释
    # plt.legend()
    plt.show()


def distribute(center, nodes, K, demand, d_limit):
    '''
    将节节点分给最近的中心
    输入：center-中心,nodes-节点,K-类数量,demand-需求,d_limit-一个簇的满足需求的能力
    输出：新的簇-clusters，簇的需求-clusters_d
    '''
    clusters = [[] for i in range(K)]  # 簇
    label = [None for i in range(len(nodes))]  # 点属于哪一簇
    clusters_d = [0 for i in range(K)]  # 簇需求
    dis_matrix = calDistance(center, nodes).astype('float64')
    for i in range(len(dis_matrix)):
        row, col = dis_matrix.stack().idxmin()
        node = nodes[row]
        j = 1
        while clusters_d[col] + demand[row] > d_limit:  # 检验约束
            if j < K:
                j += 1
                dis_matrix.loc[row, col] = math.pow(10, 10)  # 将距离设为10的10次方
                col = dis_matrix.loc[row, :].idxmin()
            else:  # 所有类都不满足需求量约束
                print("K较小")
                scatter_diagram(clusters, nodes)
                return None
        # 满足约束正常分配
        clusters[col].append(node)
        label[row] = col
        clusters_d[col] += demand[row]
        dis_matrix.loc[row, :] = math.pow(10, 10)
    return clusters, clusters_d, label
def cal_center(clusters):
    '''
    计算簇的中心
    输入：clusters-类坐标；
    输出：簇中心-new_center
    '''
    new_center = []
    for cluster in clusters:
        x, y = [], []
        for Coordinate in cluster:
            x.append(Coordinate[0])
            y.append(Coordinate[1])
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        new_center.append((round(x_mean, 2), round(y_mean, 2)))  # 四舍五入到小数点后两位
    return new_center


if __name__ == "__main__":
    # 输入数据
    nodes = [(11, 36), (13, 35), (14, 12), (25, 22), (15, 11), (16, 24), (14, 30), (16, 32), (26, 10), (27, 31),
             (22, 29), (1, 29), (20, 29), (17, 15), (5, 38),
             (3, 15), (22, 20), (4, 13), (4, 22), (12, 18), (23, 12), (18, 15), (5, 13), (9, 27), (17, 36), (2, 14),
             (1, 16), (7, 15), (25, 15), (19, 26), (25, 40),
             (26, 34), (25, 35), (2, 36), (29, 24), (17, 17), (8, 26), (4, 14), (5, 25), (6, 37), (1, 14), (6, 39),
             (11, 13), (10, 20), (21, 11), (5, 19), (5, 35), (1, 34),
             (16, 39), (19, 24), (39, 31), (49, 31), (41, 50), (31, 33), (32, 40), (35, 30), (31, 39), (34, 48),
             (42, 32), (32, 35), (35, 33), (35, 34), (43, 41), (35, 47),
             (49, 36), (37, 41), (43, 46), (41, 41), (45, 50), (41, 35), (45, 44), (41, 30), (43, 33), (31, 45),
             (48, 32), (39, 49), (38, 42), (33, 39), (49, 33), (43, 44),
             (32, 30), (40, 47), (36, 46), (47, 47), (37, 33), (35, 31), (42, 38), (43, 47), (30, 47), (30, 30),
             (37, 34), (41, 45), (27, 33), (42, 39), (43, 43), (50, 43),
             (28, 40), (35, 41), (32, 41), (31, 30)]
    demand = [3, 3, 5, 9, 6, 1, 4, 8, 8, 3, 6, 5, 6, 4, 1, 4, 7, 1, 3, 6, 2, 5, 7, 2, 2, 1, 9, 3, 7, 8, 4, 3, 1, 2, 5,
              1, 3, 4, 8, 5, 2, 9, 3, 10, 6, 1, 9, 3, 5, 3, 3, 3, 9, 9, 6, 2, 5, 2, 4,
              10, 4, 10, 10, 6, 3, 7, 9, 4, 2, 9, 7, 4, 5, 3, 3, 8, 2, 3, 2, 9, 1, 3, 3, 3, 9, 5, 7, 8, 8, 5, 2, 8, 3,
              4, 2, 4, 6, 3, 7, 4]

    print("显示初始分布图...")
    scatter_diagram([list(nodes)], nodes)  # 初始位置分布图

    # 参数
    d_limit = 150  # 一个簇最大能容纳的
    K = math.ceil(sum(demand) / d_limit)  # 簇的最小边界
    # 历史最优参数
    best_s = -1
    best_center = 0
    best_clusters = 0
    best_labels = 0
    best_clusters_d = 0

    print("开始聚类计算...")
    for n in range(20):  # 多次遍历，kmeans对初始解敏感
        # 初始化随机生成簇中心
        index = random.sample(list(range(len(nodes))), K)
        new_center = [nodes[i] for i in index]  # 从nodes中随机选取K个点作为新的中心点
        i = 1
        center_list = []
        while True:
            # 节点——> 簇
            center = new_center.copy()
            center_list.append(center)  # 保留中心，避免出现A生成B，B生成C，C有生成A的情况
            result = distribute(center, nodes, K, demand, d_limit)
            if result is None:
                # 如果K太小导致分配失败，跳过本次循环
                break
            clusters, cluster_d, label = result
            new_center = cal_center(clusters)
            if (center == new_center) | (new_center in center_list):
                break
            i += 1

        # 确保label是有效的（非None），才计算轮廓系数
        if result is not None:
            try:
                s = silhouette_score(nodes, label, metric='euclidean')  # 计算轮廓系数，聚类的评估指标
                if best_s < s:
                    best_s = s
                    best_clusters_d = cluster_d
                    best_center = center.copy()
                    best_clusters = clusters.copy()
                    best_labels = label.copy()
            except Exception as e:
                pass
    print("显示聚类结果图...")
    scatter_diagram(best_clusters, nodes)  # 历史最优图
    # print("各类需求量：",best_clusters_d)