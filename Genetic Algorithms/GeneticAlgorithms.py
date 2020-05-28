# @Time : 2020/5/27 22:33
# @Author : jingwei
# @Site : https://github.com/jingwei1205
# @File : GeneticAlgorithms.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

# 声明全局变量
global sca


class Population:
    def __init__(self, DNA_length, population_size, cross_rate, mutation_rate, generations, x):
        """构造种族的基本信息
        :param DNA_length: 染色体编码长度
        :param population_size: 种族数量
        :param cross_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param generations: 迭代次数
        :param x: x取值范围
        :return: null
        """
        self._DNA_length = DNA_length
        self._population_size = population_size
        self._cross_rate = cross_rate
        self._mutation_rate = mutation_rate
        self._generations = generations
        self._x = x
        pass

    @staticmethod
    def get_fitness(value):
        """
        :param value: 传入函数值
        :return: 返回处理过的适应值，计算适应度,因为为求函数最大值，所以可以直接使用函数值的大小代替适应度。
        同时考虑到函数值可能为负数，后续使用轮盘赌算法淘汰不好的基因会发生问题，所以减去这次子代的最小的值，
        就可以使所有的函数值为正，同时加上一个很小的值，防止出现轮盘赌概率为0的情况。
        """
        return value - np.min(value) + 1e-4

    @staticmethod
    def Function(x):
        """
        :param x:传入x横坐标
        :return: 返回函数值
        """
        return np.sin(4 * x) * x + 1.1 * np.sin(4 * x + 2) + 8 * np.sin(x - 2) + 0.7 * np.sin(12 * x - 4)

    def select(self, pop, fitness):
        """选择
        :param pop:种群基因数组
        :param fitness: 适应度
        :return:被选择的基因
        """
        index = np.random.choice(np.arange(self._population_size),
                                 size=self._population_size,
                                 replace=True,
                                 p=fitness / fitness.sum())
        return pop[index]

    def cross(self, parent, pop):
        """交叉
        :param parent:父辈基因
        :param pop: 种群基因里
        :return: 交叉后的基因，即子辈基因
        """
        if np.random.rand() < self._cross_rate:
            i = np.random.randint(0, self._population_size)
            cross_points = np.random.randint(0, 2, size=self._DNA_length)
            parent[cross_points] = pop[i, cross_points]
        return parent

    def mutation(self, child):
        """变异
        :param child:孩子基因
        :return: 变异或非变异孩子基因
        """
        for index in range(self._DNA_length):
            if np.random.rand() < self._mutation_rate:
                child[index] = not child[index]
        return child

    def translate(self, pop):
        """
        :param pop:二进制基因编码
        :return: 十进制
        """
        return pop.dot(2 ** np.arange(self._DNA_length)[::-1]) / float(2 ** self._DNA_length - 1) * (
                    self._x[1] - self._x[0])

    # 遗传算法实现方法
    def start_genetic_algorithms(self):
        # 定义全局变量sca，为动态点消失做条件准备
        global sca
        # 随机生成值为0或1组成的（种群数量*基因序列长度的）矩阵
        pop = np.random.randint(2, size=(self._population_size, self._DNA_length))
        # 生成动态图，需要在show的时候前使用ioff不然会一闪而过的错误
        plt.ion()
        # 将区间拆分成5万等分
        x = np.linspace(*self._x, 50000)
        # 画出函数图像
        plt.plot(x, Population.Function(x))
        # 迭代繁衍，到达规定迭代次数后退出
        for i in range(self._generations):
            # 获取转码后的x对应的函数值
            values = Population.Function(self.translate(pop))
            # 如果sca在全局变量里，就移除，这里代表红色的点动态消失，间隔为0.03秒，方便人观察遗传算法的学习的过程
            if 'sca' in globals():
                sca.remove()
            sca = plt.scatter(self.translate(pop), values, s=100, lw=0, c='red', alpha=0.5)
            plt.pause(0.03)
            # 得到适应值
            fitness = Population.get_fitness(values)
            # 输出迭代最好基因，即适应值最大的基因
            good_gene = pop[np.argmax(fitness), :]
            print("第", i + 1, "代最合适的DNA", good_gene,
                  "\n横坐标为", self.translate(good_gene), "函数值为", Population.Function(self.translate(good_gene)))
            # 轮盘赌算法，按照适应值比例淘汰不好的样本
            pop = self.select(pop, fitness)
            # 复制此时种群基因副本
            pop_copy = pop.copy()
            # 开始繁衍后代
            for parent in pop:
                # 父辈交叉生出子辈
                child = self.cross(parent, pop_copy)
                # 孩子变异或非变异
                child = self.mutation(child)
                # 孩子变成父辈，深复制，避免空间浪费，指针还是指向原来的parent指向的地方，只是内容改变为child
                parent[:] = child
        plt.ioff()
        plt.show()
        pass


if __name__ == "__main__":
    """
    求解自定义函数F(x)=np.sin(4 * x * 3) * x + 1.1 * np.sin(4 * x + 2) + 8 * np.sin(x - 2) + 0.7 * np.sin(12 * x - 4)
    创建种群对象，赋值dna长度，种群个数，交叉概率，变异概率，迭代次数
    假设精确到四位小数，x取值范围[0,5]，那么(5-0)*1e+4=5e+4个等分,所以取16位编码
    """
    population = Population(DNA_length=15,
                            population_size=100,
                            cross_rate=0.9,
                            mutation_rate=0.002,
                            generations=300,
                            x=[0, 5])
    # 调用对象遗传算法方法
    population.start_genetic_algorithms()
    pass
