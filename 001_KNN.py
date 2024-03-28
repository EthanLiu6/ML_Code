"""
    实现KNN算法(二维)
"""
import math


class KNN:
    def __init__(self):
        # data[特征列1, 特征列2, 标签]
        data = [
            [1, 2, 1],
            [1.3, 2.3, 1],
            [0.6, -1, 1],
            [4, -3, -1],
            [2, 1, 1],
            [5, -3.6, -1],
            [4.6, -4, -1],
            [1.2, 1.75, 1],
        ]
        self.data = data

        self.target = [1.8, 1.8]  # 预测值

    def calculateDistance(self):
        # 特征向量
        X = [row[:2] for row in self.data]  # 特征矩阵
        y = [row[-1] for row in self.data]  # 标签列向量
        dist = []  # 记录预测数据与真实数据的欧几里得距离(外加对应标签)
        i = 0  # 为方便定位对应标签
        for x in X:
            dis = math.sqrt(math.pow(x[0] - self.target[0], 2) + math.pow(x[1] - self.target[1], 2))
            dist.append([dis, y[i]])
            i += 1
        return sorted(dist)  # 对计算的距离排序(小到大<-->近到远)

    def kNeighbour(self, k):  # 指定k个邻居
        reses = self.calculateDistance()
        neighbour = 0  # 邻居计数器
        predict = 0  # 标签为-1和1, 所以预测值可以先直接连续加(判断正负即可确定类别)
        for res in reses:
            if neighbour == k:  # 找到k个邻居即可
                break
            predict += res[1]  # 先直接连续加
            neighbour += 1
        self.target.append(1 if predict > 0 else -1)  # 判断正负即可确定类别
        return self.target


if __name__ == '__main__':
    k1 = KNN()
    print(k1.kNeighbour(5))
