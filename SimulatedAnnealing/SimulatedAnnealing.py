# @Time : 2020/5/30 22:45
# @Author : jingwei
# @Site : https://github.com/jingwei1205
# @File : SimulatedAnnealing.py
# @Software: PyCharm

import math
import random
import numpy as np
import matplotlib.pyplot as plt
global sca


class SimulatedAnnealing:
    def __init__(self, scope, temp, end_temp, speed):
        """初始化基本参数
        :param scope: 定义域
        :param temp: 开始温度
        :param end_temp: 结束温度
        :param speed: 降温系数
        :return: null
        """
        self.scope = scope
        self.temp = temp
        self.end_temp = end_temp
        self.speed = speed
        self.count = 0
        pass

    @staticmethod
    def Function(x):
        """
        :param x:传入x横坐标
        :return: 返回函数值
        """
        return np.sin(4 * x) * x + 1.1 * np.sin(4 * x + 2) + 8 * np.sin(x - 2) + 0.7 * np.sin(12 * x - 4)

    def max_judge(self, de):
        """搜索最大值时使用：判断新值是否要大于旧值，如果大于就接受，否则以一定概率接受
        :param de: 新旧差值
        :return: 是否采用
        """
        if de > 0:
            return True
        else:
            if math.exp(-self.speed/self.temp) > random.random():
                return True
            else:
                return False
        pass

    def start_simulated_annealing(self):
        global sca
        x_old = (self.scope[1] - self.scope[0]) * random.random() + self.scope[0]
        y_old = SimulatedAnnealing.Function(x_old)
        # 生成动态图，需要在show的时候前使用ioff不然会一闪而过的错误
        plt.ion()
        # 将区间拆分成5万等分
        x = np.linspace(*self.scope, 50000)
        # 画出函数图像
        plt.plot(x, SimulatedAnnealing.Function(x))
        while self.temp > self.end_temp:
            # x随机小范围跳动
            delta = (random.random() - 0.5) * 3
            x_new = x_old + delta
            # 当扰动的值超过定义域时，进行重新操作重新变回定义域内
            if x_new > self.scope[1] or x_new < self.scope[0]:
                x_new = x_new - 2 * delta
            y_new = SimulatedAnnealing.Function(x_new)
            de = y_new-y_old
            # 判断是否运用变化
            checkout = self.max_judge(de)
            if checkout:
                y_old = y_new
                x_old = x_new
            # 如果新的最优值出现就降温
            if de > 0:
                if 'sca' in globals():
                    sca.remove()
                sca = plt.scatter(x_old, y_old, s=100, lw=0, c='red', alpha=0.5)
                plt.pause(0.03)
                self.temp *= self.speed
            else:
                self.count += 1
            # 长时间找不到更优解或者跳动概率较低，那么退出循环了
            if self.count > 5000:
                break
        print("当前x值", x_old, "当前最大值", y_old)
        pass


if __name__ == "__main__":
    simulated_annealing = SimulatedAnnealing(scope=[0, 5], temp=1e5, end_temp=1e-3, speed=0.98)
    simulated_annealing.start_simulated_annealing()
