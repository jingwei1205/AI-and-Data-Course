# @Time    : 2020/5/18 15:21
# @Author  : jingwei
# @Site : https://github.com/jingwei1205
# @FileName: K-Mean.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


# K-Means算法类
class KMeanAlgorithm:
    def __init__(self, k, file, init):
        """
            :param k:分簇个数
            :param file:数据集文件名，如果文件名为none，则对象自己通过函数会给出一个数据集，方便老师的运行
            :param init:初始化聚类中心
        """
        self._K = k
        self._file_name = file
        self._cluster_group = []
        self._init_group = init
        self._change_center = True

    # 读取数据集
    def read_data(self):
        results = []
        # 如果传入对象的文件名变量为none，默认给出一个数据集，方便老师的运行
        if self._file_name == "none":
            data = [
                [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
                [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
                [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
                [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
                [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]
            ]
            return np.array(data)
        with open(self._file_name, "r", encoding="utf-8") as data_file:
            for line in data_file.readlines():
                data = line.strip().split()
                results.append([np.float(data[0]), np.float(data[1])])
            return np.array(results)

    # 选择初始化聚类中心
    def init_cluster_centers(self):
        return np.array(self._init_group)

    # 实现
    def start_algorithm(self):
        # 获取数据集
        data_array = self.read_data()
        # 获取初始化聚类中心
        cluster_center = self.init_cluster_centers()
        # 为储存K类序号的数组初始化
        for _ in range(self._K):
            self._cluster_group.append([])
        # 如果聚类中心发生改变就一直循环
        while self._change_center:
            # 使用媒介存放上个聚类中心
            temp = cluster_center
            print("temp:")
            print(temp)
            # 假设聚类中心不改变
            self._change_center = not self._change_center
            # 将分类数组初始化为空
            for i in range(self._K):
                self._cluster_group[i] = []
            # 数组数据有几行就循环几次
            for i in range(data_array.shape[0]):
                # 获取这个点到三个聚类中心的距离
                distances = np.sum((cluster_center - data_array[i]) ** 2, axis=1)
                print(distances)
                # 使用函数找出最小的值的点索引，把这个点序号加到哪个聚类中
                index = np.argmin(distances)
                self._cluster_group[int(index)].append(i)
                print(self._cluster_group)
            for i in range(self._K):
                cluster_center[i] = np.sum(data_array[self._cluster_group[i]], axis=0) / len(self._cluster_group[i])
            if not (temp == cluster_center).all():
                self._change_center = True
        # 用小点表示数据集坐标
        for i in range(self._K):
            plt.scatter(data_array[self._cluster_group[i], 0], data_array[self._cluster_group[i], 1])
        # 用“+”表示聚类中心
        plt.scatter(cluster_center[:, 0], cluster_center[:, 1], s=100, c='r', marker="+")
        # 画图
        plt.show()


if __name__ == '__main__':
    # KM = KMeanAlgorithm(3, "KMeansData.txt", [[0.343, 0.099], [0.719, 0.103], [0.751, 0.489]])
    # 创建KMeanAlgorithm类对象KM，传入K=3，分三类，"none"表示使用
    KM = KMeanAlgorithm(k=3, file="none", init=[[0.343, 0.099], [0.719, 0.103], [0.751, 0.489]])
    KM.start_algorithm()
