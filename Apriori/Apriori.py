import numpy as np
def loadDataSet():
  return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dateSet):
  c1 = []
  for line in dateSet:
    for item in line:
      if not [item] in c1:
        c1.append([item])
  c1.sort()
  return list(map(frozenset,c1))

def scanData(data,ck,minSupport):#寻找满足最小支持度的项集
  ssCnt = {}
  for tid in data:
    for can in ck:
      if can.issubset(tid):
        if can not in ssCnt.keys():
          ssCnt[can] = 0
        ssCnt[can] += 1
  numItems = len(data)
  retList = []
  supportData = {}
  for key in ssCnt.keys():
    support = ssCnt[key]/numItems
    if support >= minSupport:
      retList.append(key)
    supportData[key] = support
  return retList,supportData


def aprioriGen(Lk,k): #根据k-1项集生成k项集
  retList = []
  lenLk = len(Lk)
  for i in range(lenLk):
    for j in range(i+1,lenLk):
      l1 = list(Lk[i])[:k-2]
      l2 = list(Lk[j])[:k-2]
      l1.sort()
      l2.sort()
      if l1 == l2:
        retList.append(Lk[i] | Lk[j])
  return retList

def apriori(dataSet,minSupport = 0.5):#生成频繁项集
  c1 = createC1(dataSet)
  D = list(map(set,dataSet))
  l1,supportData = scanData(D,c1,minSupport)
  L = [l1]
  k = 2
  while(len(L[k-2])>0):
    ck = aprioriGen(L[k-2],k)
    lk,supk = scanData(D,ck,minSupport)
    k = k + 1
    L.append(lk)
    supportData.update(supk)
  return L,supportData
def generaterRules(L,supportData,minConf=0.7):#生成规则
  bigRuleList = []
  for i in range(1,len(L)):
    for freqSet in L[i]:
      H1 = [frozenset([item]) for item in freqSet]
      if i>1:
        rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
      else:
        calcConf(freqSet,H1,supportData,bigRuleList,minConf)
  return bigRuleList
def calcConf(freqSet,H,suppurtData,brl,minConf = 0.7):#计算满足置信度的规则
  prunedH = []
  for conseq in H:
    conf = suppurtData[freqSet]/suppurtData[freqSet-conseq]
    if conf > minConf:
      brl.append((freqSet-conseq,conseq,conf))
      prunedH.append(conseq)
  return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):#递归生成规则
  m = len(H[0])
  if len(freqSet)>=(m+1):
    Hmp1 = calcConf(freqSet,H,supportData,brl,minConf)
    if (len(Hmp1) > 1):
      Hmp1 = aprioriGen(Hmp1,m+1)
      rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)




data = [line.split() for line in open('mushroom.dat').readlines()]
L,support = apriori(data,minSupport=0.3)
for i in range(len(L)):
  for item in L[i]:
    if item & {'2'}:
      print(item)
