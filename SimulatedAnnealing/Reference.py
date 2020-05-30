# @Time : 2020/5/30 22:36
# @Author : jingwei
# @Site : https://github.com/jingwei1205
# @File : Reference.py
# @Software: PyCharm

from matplotlib import pyplot as plt
import numpy as np


def coordinate_init(size):
    # 产生坐标字典
    coordinate_dict = {}
    coordinate_dict[0] = (0, 0)  # 起点是（0，0）
    for i in range(1, size + 1):  # 顺序标号随机坐标
        coordinate_dict[i] = (np.random.uniform(0, size), np.random.uniform(0, size))
    coordinate_dict[size + 1] = (0, 0)  # 终点是（0,0)
    return coordinate_dict


def distance_matrix(coordinate_dict, size):  # 生成距离矩阵
    d = np.zeros((size + 2, size + 2))
    for i in range(size + 1):
        for j in range(size + 1):
            if (i == j):
                continue
            if (d[i][j] != 0):
                continue
            x1 = coordinate_dict[i][0]
            y1 = coordinate_dict[i][1]
            x2 = coordinate_dict[j][0]
            y2 = coordinate_dict[j][1]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if (i == 0):
                d[i][j] = d[size + 1][j] = d[j][i] = d[j][size + 1] = distance
            else:
                d[i][j] = d[j][i] = distance
    return d


def path_length(d_matrix, path_list, size):  # 计算路径长度
    length = 0
    for i in range(size + 1):
        length += d_matrix[path_list[i]][path_list[i + 1]]
    return length


def new_path(path_list, size):
    # 二交换法
    change_head = np.random.randint(1, size + 1)
    change_tail = np.random.randint(1, size + 1)
    if (change_head > change_tail):
        change_head, change_tail = change_tail, change_head
    change_list = path_list[change_head:change_tail + 1]
    change_list.reverse()  # change_head与change_tail之间的路径反序
    new_path_list = path_list[:change_head] + change_list + path_list[change_tail + 1:]
    return change_head, change_tail, new_path_list


def diff_old_new(d_matrix, path_list, new_path_list, head, tail):  # 计算新旧路径的长度之差
    old_length = d_matrix[path_list[head - 1]][path_list[head]] + d_matrix[path_list[tail]][path_list[tail + 1]]
    new_length = d_matrix[new_path_list[head - 1]][new_path_list[head]] + d_matrix[new_path_list[tail]][
        new_path_list[tail + 1]]
    delta_p = new_length - old_length
    return delta_p


T_start = 2000  # 起始温度
T_end = 1e-20  # 结束温度
a = 0.995  # 降温速率
Lk = 50  # 内循环次数,马尔科夫链长
size = 20
coordinate_dict = coordinate_init(size)
print(coordinate_dict)  # 打印坐标字典
path_list = list(range(size + 2))  # 初始化路径
d = distance_matrix(coordinate_dict, size)  # 距离矩阵的生成
best_path = path_length(d, path_list, size)  # 初始化最好路径长度
print('初始路径:', path_list)
print('初始路径长度:', best_path)
best_path_temp = []  # 记录每个温度下最好路径长度
best_path_list = []  # 用于记录历史上最好路径
balanced_path_list = path_list  # 记录每个温度下的平衡路径
balenced_path_temp = []  # 记录每个温度下平衡路径(局部最优)的长度
while T_start > T_end:
    for i in range(Lk):
        head, tail, new_path_list = new_path(path_list, size)
        delta_p = diff_old_new(d, path_list, new_path_list, head, tail)
        if delta_p < 0:  # 接受状态
            balanced_path_list = path_list = new_path_list
            new_len = path_length(d, path_list, size)
            if (new_len < best_path):
                best_path = new_len
                best_path_list = path_list
        elif np.random.random() < np.exp(-delta_p / T_start):  # 以概率接受状态
            path_list = new_path_list
    path_list = balanced_path_list  # 继承该温度下的平衡状态（局部最优）
    T_start *= a  # 退火
    best_path_temp.append(best_path)
    balenced_path_temp.append(path_length(d, balanced_path_list, size))
print('结束温度的局部最优路径:', balanced_path_list)
print('结束温度的局部最优路径长度:', path_length(d, balanced_path_list, size))
print('最好路径:', best_path_list)
print('最好路径长度:', best_path)
x = []
y = []
for point in best_path_list:
    x.append(coordinate_dict[point][0])
    y.append(coordinate_dict[point][1])
plt.figure(1)
plt.subplot(311)
plt.plot(balenced_path_temp)  # 每个温度下平衡路径长度
plt.subplot(312)
plt.plot(best_path_temp)  # 每个温度下最好路径长度
plt.subplot(313)
plt.scatter(x, y)
plt.plot(x, y)  # 路径图
plt.grid()
plt.show()
