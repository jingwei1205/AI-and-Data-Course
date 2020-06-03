# @Time : 2020/6/2 23:20
# @Author : jingwei
# @Site : 
# @File : DecisionTree.py
# @Software: PyCharm

from math import log
import operator


class DecisionTree:
    def __init__(self, data, labels):
        """
        :param data: 数据集
        :param labels: 数据集标签
        """
        self.data = data
        self.labels = labels

    def calculate_data(self, dataSet):
        """
        :param dataSet:数据集
        :return: 信息熵
        """
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]  # 每行数据的最后一个字（类别）
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1  # 统计有多少个类以及每个类的数量
        shannonEnt = 0
        for key in labelCounts:
            # 计算单个熵值
            prob = float(labelCounts[key]) / numEntries
            # 累加熵值
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def get_data(self):  # 创造示例数据
        return self.data, self.labels

    def classfy_data(self, dataSet, axis, value):  # 按某个特征分类后的数据
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def choose_best(self, dataSet):  # 选择最优的分类特征
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calculate_data(dataSet)  # 原始的熵
        bestInfoGain = 0
        bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0
            for value in uniqueVals:
                subDataSet = self.classfy_data(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calculate_data(subDataSet)  # 按特征分类后的熵
            infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
            if infoGain > bestInfoGain:  # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def sort(self, classList):  # 按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def create_tree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.sort(classList)
        bestFeat = self.choose_best(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del (labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.create_tree(self.classfy_data(dataSet, bestFeat, value), subLabels)
        return myTree


if __name__ == '__main__':
    decision = DecisionTree([['本科', '有经验', '录取'], ['专科', '有经验', '录取'], ['本科', '无经验', '录取'], ['专科', '无经验', '拒录']], ['学历', '经验'])
    dataSet, labels = decision.get_data()
    # 输入决策树的模型
    print(decision.create_tree(dataSet, labels))
