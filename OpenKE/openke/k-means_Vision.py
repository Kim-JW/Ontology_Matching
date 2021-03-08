import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

array = []
centNum = 10
for line in open("/home/syx/Desktop/datasets/train_hash.txt", encoding='utf-8', mode='r'):
    sub, pred, obj = line.split(',')
    array.append([sub, pred, obj])

array = np.array(array).astype(float)

'''
clf = mixture.GaussianMixture(n_components=centNum, covariance_type='diag', max_iter=300,
                              init_params='random').fit(array)
pre = clf.fit_predict(array)
'''

clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
             n_clusters=centNum, n_init=10, n_jobs=1, precompute_distances='auto',
             random_state=None, tol=0.0001, verbose=0)
pre = clf.fit_predict(array)

print(type(clf.cluster_centers_))
fileObject = open('/home/syx/Desktop/datasets/2pi/Hash_Triple2HashValues_100_2PI.txt', 'w')
fileObject.write(str(clf.cluster_centers_))
fileObject.write('\n')
for i in range(centNum):
    subject = array[pre == i, 0]
    predicate = array[pre == i, 1]
    object = array[pre == i, 2]
    list = []
    for j in range(len(subject)):
        list.append([subject[j], predicate[j], object[j]])
    fileObject.write(str(list))
    fileObject.write('\n')
    centroide = [sum(subject)/len(subject),sum(predicate)/len(predicate),sum(object)/len(object)]
    print(centroide)

fileObject.close()
'''

#visionlization
colors = ['#1f77b4', '#ff7f0e', '#8c564b','#06623B','#2ca02c', '#d62728', '#9467bd',  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
fig = plt.figure()
ax = Axes3D(fig)
'''
'''
for i in range(centNum):
    ax.scatter(array[pre == i, 0], array[pre == i, 1], array[pre == i, 2], s=15, c=colors[i], marker='o',
               edgecolor='black',
               label='cluster ' + str(i))
'''
'''
# plot the centroids
ax.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], clf.cluster_centers_[:, 2], s=50, marker='*',
           c='red', edgecolor='black',
           label='centroid')

ax.legend(scatterpoints=1)

# ax.title('K-Means')

ax.set_title('TransH')

ax.set_xlabel('subject')
ax.set_ylabel('predicate')
ax.set_zlabel('object')
# plt.grid()
plt.show()

'''
