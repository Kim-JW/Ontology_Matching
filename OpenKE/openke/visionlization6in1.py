# Coding=utf-8
# Version:python3.6.0
# Tools:Pycharm 2017.3.2
_data_ = '4/27/2020 10:48 PM'
_author_ = 'SUN'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data_x = []
data_y = []
data_z = []
for line in open("/home/syx/Desktop/datasets/Hash_Triple2HashValues_100.txt"):
    line = line.replace("\n", "")
    data_x.append(float(line.split(',')[0]))
    data_y.append(float(line.split(',')[1]))
    data_z.append(float(line.split(',')[2]))

#data = np.random.randint(0, 255, size=[40, 40, 40])
#print(data.shape)
x, y, z = data_x, data_y, data_z
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[:272115], y[:272115], z[:272115], marker=".", c='b', s=0.1)  # 绘制数据点
#ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
#ax.scatter(x[30:40], y[30:40], z[30:40], c='g')

ax.set_title('Hash Function')

ax.set_xlabel('subject')
ax.set_ylabel('predicate')
ax.set_zlabel('object')  # 坐标轴










plt.show()
