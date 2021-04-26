import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

da = []
'''
for line in open(r'C:/Users/syx92/Google Drive/Linked Discover/Triple2Vector/mouse/ConvE_mouse_Triple2Vector.txt'):
    da.append([float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])])
'''

for line in open("human/transd_human_Triple2Vector.txt"):
    da.append([float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])])

data = np.array(da)

model = DBSCAN(eps=0.04, min_samples=1)
model.fit_predict(data)
pred = model.fit_predict(data)

print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)
centroid = []
for i in range(len(set(model.labels_))):
    centroid_0 = []
    centroid_1 = []
    centroid_2 = []
    for idx, val in enumerate(model.labels_ == i):
        if val:
            centroid_0.append(data[idx][0])
            centroid_1.append(data[idx][1])
            centroid_2.append(data[idx][2])
    centroid.append([np.mean(centroid_0), np.mean(centroid_1), np.mean(centroid_2)])
    centroid_0.clear()
    centroid_1.clear()
    centroid_2.clear()

print(centroid)

cent = np.array(centroid)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=model.labels_, s=5)
ax.scatter(cent[:, 0], cent[:, 1], cent[:, 2], s=20, marker='*', c='red', label='centroid')
ax.view_init(azim=300)
ax.set_xlabel('subject')
ax.set_ylabel('predicate')
ax.set_zlabel('object')
plt.legend(loc='upper left')
plt.title('Sample graph')
plt.show()
