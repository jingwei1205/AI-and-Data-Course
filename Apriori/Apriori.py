# @Time : 2020/5/27 22:33
# @Author : jingwei
# @Site : https://github.com/jingwei1205
# @File : GeneticAlgorithms.py
# @Software: PyCharm

from numpy import *


class Apriori:
    def __init__(self, data):
        """
        :param data: 用户给定的数据集
        """
        self.data = data

    # 将元素转换为frozenset型字典，集合不能添加和删除操作
    def frozen(self):
        c1 = []
        for transaction in self.data:
            for item in transaction:
                if not [item] in c1:
                    c1.append([item])
        c1.sort()
        # 为了后面可以将这些值作为字典的键使用frozenset
        return list(map(frozenset, c1))

    # 过滤掉不符合支持度的集合
    # 返回 频繁项集列表retList 所有元素的支持度字典
    @staticmethod
    def scan(D, Ck, minSupport):
        ssCnt = {}
        for tid in D:
            for can in Ck:
                if can.issubset(tid):  # 判断can是否是tid的《子集》 （这里使用子集的方式来判断两者的关系）
                    if can not in ssCnt:  # 统计该值在整个记录中满足子集的次数（以字典的形式记录，frozenset为键）
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1
        numItems = float(len(D))
        retList = []  # 重新记录满足条件的数据值（即支持度大于阈值的数据）
        supportData = {}  # 每个数据值的支持度
        for key in ssCnt:
            support = ssCnt[key] / numItems
            if support >= minSupport:
                retList.insert(0, key)
            supportData[key] = support
        return retList, supportData  # 排除不符合支持度元素后的元素 每个元素支持度

    # 生成所有可以组合的集合
    # 频繁项集列表Lk 项集元素个数k  [frozenset({2, 3}), frozenset({3, 5})] -> [frozenset({2, 3, 5})]
    def apriori_gen(self, Lk, k):
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk):  # 两层循环比较Lk中的每个元素与其它元素
            for j in range(i + 1, lenLk):
                L1 = list(Lk[i])[:k - 2]  # 将集合转为list后取值
                L2 = list(Lk[j])[:k - 2]
                L1.sort()
                L2.sort()  # 这里说明一下：该函数每次比较两个list的前k-2个元素，如果相同则求并集得到k个元素的集合
                if L1 == L2:
                    retList.append(Lk[i] | Lk[j])  # 求并集
        return retList  # 返回频繁项集列表Ck

    # 封装所有步骤的函数
    # 返回 所有满足大于阈值的组合 集合支持度列表
    def apriori(self, minSupport=0.5):
        D = list(map(set, self.data))  # 转换列表记录为字典
        C1 = self.frozen()  # 将每个元素转会为frozenset字典
        L1, supportData = Apriori.scan(D, C1, minSupport)  # 过滤数据
        L = [L1]
        k = 2
        while (len(L[k - 2]) > 0):  # 若仍有满足支持度的集合则继续做关联分析
            Ck = self.apriori_gen(L[k - 2], k)  # Ck候选频繁项集
            Lk, supK = Apriori.scan(D, Ck, minSupport)  # Lk频繁项集
            supportData.update(supK)  # 更新字典（把新出现的集合:支持度加入到supportData中）
            L.append(Lk)
            k += 1  # 每次新组合的元素都只增加了一个，所以k也+1（k表示元素个数）
        return L, supportData

    # 获取关联规则的封装函数
    def generate_rules(self, L, supportData, minConf=0.7):  # supportData 是一个字典
        bigRuleList = []
        for i in range(1, len(L)):  # 从为2个元素的集合开始
            for freqSet in L[i]:
                # 只包含单个元素的集合列表
                H1 = [frozenset([item]) for item in freqSet]  # frozenset({2, 3}) 转换为 [frozenset({2}), frozenset({3})]
                # 如果集合元素大于2个，则需要处理才能获得规则
                if (i > 1):
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)  # 集合元素 集合拆分后的列表 。。。
                else:
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
        return bigRuleList

    # 对规则进行评估 获得满足最小可信度的关联规则
    @staticmethod
    def calcConf(freqSet, H, supportData, brl, minConf=0.7):
        prunedH = []  # 创建一个新的列表去返回
        for conseq in H:
            conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算置信度
            if conf >= minConf:
                print(freqSet - conseq, '-->', conseq, 'conf:', conf)
                brl.append((freqSet - conseq, conseq, conf))
                prunedH.append(conseq)
        return prunedH

    # 生成候选规则集合
    def rulesFromConseq(self, freqSet, H, supportData, brl, minConf=0.7):
        m = len(H[0])
        if (len(freqSet) > (m + 1)):  # 尝试进一步合并
            Hmp1 = self.apriori_gen(H, m + 1)  # 将单个集合元素两两合并
            Hmp1 = Apriori.calcConf(freqSet, Hmp1, supportData, brl, minConf)
            if (len(Hmp1) > 1):  # need at least two sets to merge
                self.rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


if __name__ == "__main__":
    apriori = Apriori([[1, 3], [2, 3, 4], [1, 2, 3, 4, 5], [2, 5]])
    L, suppData = apriori.apriori()
    rules = apriori.generate_rules(L, suppData, minConf=0.7)
    print("频繁项集如下：")
    for key in L:
        print(key)
    print("各支持度如下：")
    for key in suppData:
        print(key, suppData[key])
