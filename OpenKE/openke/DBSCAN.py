import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

size = 30


##计算欧式距离
def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


##随机产生n个dim维度的数据 (这里为了展示结果 dim取2或者3)
def genDataset(n, dim):
    data = []
    while len(data) < n:
        p = np.around(np.random.rand(dim) * size, decimals=2)
        data.append(p)
    return data


##判断两点是否在范围内
def isNeighbor(x, y, eps):
    return distEuclid(x, y) <= eps


##获取某一点邻域内的点
def getSeedPos(pos, data, eps):
    seed = []
    for p in range(len(data)):
        if isNeighbor(data[p], data[pos], eps):
            seed.append(p)
    return seed


##获取核心点列表
def getCorePointsPos(data, eps, minpts):
    cpoints = []
    for pos in range(len(data)):
        if len(getSeedPos(pos, data, eps)) >= minpts:
            cpoints.append(pos)
    return cpoints


##分类
def getCluster(data, eps, minpts):
    corePos = getCorePointsPos(data, eps, minpts)
    unvisited = list(range(len(data)))
    cluster = {}
    num = 0

    for pos in corePos:
        if pos not in unvisited:
            continue
        clusterpoint = []
        clusterpoint.append(pos)
        seedlist = getSeedPos(pos, data, eps)
        unvisited.remove(pos)
        while seedlist:
            p = seedlist.pop(0)
            if p not in unvisited:
                continue
            unvisited.remove(p)
            clusterpoint.append(p)
            if p in corePos:
                seedlist.extend(getSeedPos(p, data, eps))
        cluster[num] = clusterpoint
        num += 1
    cluster["noisy"] = unvisited
    return cluster


##展示结果  各类簇使用不同的颜色  中心点使用X表示
def Show(data, cluster):
    num, dim = data.shape
    color = ['r', 'g', 'c', 'y', 'm', 'b', 'pink', 'maroon', 'tomato', 'peru', 'lawngreen', 'gold', 'aqua',
             'dodgerblue']
    ##二维图
    if dim == 2:
        for i in cluster:
            pos = cluster[i]
            if i == "noisy":
                for p in pos:
                    plt.plot(data[p, 0], data[p, 1], 'o', c='k')
            else:
                for p in pos:
                    plt.plot(data[p, 0], data[p, 1], 'o', c=color[i])
                    ##三维图
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in cluster:
            pos = cluster[i]
            if i == "noisy":
                for p in pos:
                    ax.scatter(data[p, 0], data[p, 1], data[p, 2], c='black')
            else:
                for p in pos:
                    ax.scatter(data[p, 0], data[p, 1], data[p, 2], c=color[i])

    plt.show()
array = []
for line in open("benchmarks/FB15K237/word2vec_skip-gram_Triple2Vector_test.txt", encoding='utf-8', mode='r'):
    sub, pred, obj = line.split(',')
    array.append([sub, pred, obj])

array = np.array(array).astype(float)
data = np.array(genDataset(80, 3))

cl = getCluster(array, 5, 3)
Show(data, cl)
