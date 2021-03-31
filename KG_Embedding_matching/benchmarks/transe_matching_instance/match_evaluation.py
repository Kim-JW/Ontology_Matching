import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

'''
f = open('benchmarks/transe_matching_instance/reference.txt', 'a')
lis = []
for line in open("./benchmarks/transe_matching_instance/transe_reference.txt"):
    if line is not '\n':
        line = line.replace("\n", "")
        lis.append(line)
lis = np.array(lis).reshape(1516, 2)
print(lis)
for lines in lis:
    f.write(str(lines)+'\n')
f.close()
'''

f = open('match_reference.txt', 'a')
mouse = []
for line in open('transe_mouse_reference_triple2vector.txt'):
    line = line.replace("\n", "")
    mouse.append(line)

human = []
for line in open('transe_human_reference_triple2vector.txt'):
    line = line.replace("\n", "")
    human.append(line)

reference = dict(zip(mouse, human))

for key, value in reference.items():
    f.write(str(key)+','+str(value)+'\n')
f.close()

# Clutering Vectors using DBScan

mouse = []
for line in open('transe_mouse_Triple2Vector.txt'):
    mouse.append([float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])])
human = []
for lines in open('transe_human_Triple2Vector.txt'):
    human.append([float(lines.split(",")[0]), float(lines.split(",")[1]), float(lines.split(",")[2])])

mouse = np.array(mouse)
human = np.array(human)

mouse_model = DBSCAN(eps=0.05, min_samples=1)
mouse_model.fit_predict(mouse)

human_model = DBSCAN(eps=0.1, min_samples=1)
human_model.fit_predict(human)

print("number of mouse cluster found: {}".format(len(set(mouse_model.labels_))))
print('cluster for each mouse point: ', mouse_model.labels_)

print("number of human cluster found: {}".format(len(set(human_model.labels_))))
print('cluster for each human point: ', human_model.labels_)

# Classify Clustering Point

for i in range(len(set(mouse_model.labels_))):
    cluster_mouse = []
    for idx, val in enumerate(mouse_model.labels_ == i):
        if val:
            cluster_mouse.append([mouse[idx][0], mouse[idx][1], mouse[idx][2]])
    f = open('mouse/'+str(i)+'.txt', 'a')
    # dic[np.mean(cluster)] = cluster
    for ins in cluster_mouse:
        f.write(str(ins[0])+','+str(ins[1])+','+str(ins[2])+'\n')
    f.close()


for i in range(len(set(human_model.labels_))):
    cluster_human = []
    for idx, val in enumerate(human_model.labels_ == i):
        if val:
            cluster_human.append([human[idx][0], human[idx][1], human[idx][2]])
    f = open('human/'+str(i)+'.txt', 'a')
    # dic[np.mean(cluster)] = cluster
    for ins in cluster_human:
        f.write(str(ins[0])+','+str(ins[1])+','+str(ins[2])+'\n')
    f.close()


# general centroid

centro_id = []
for i in range(10):
    subject = []
    predicate = []
    object = []
    for line in open('human/' + str(i) + '.txt'):
        subject.append(float(line.split(",")[0]))
        predicate.append(float(line.split(",")[1]))
        object.append(float(line.split(",")[2]))
    centro_id.append([np.mean(subject), np.mean(predicate), np.mean(object)])
    subject.clear()
    predicate.clear()
    object.clear()


print(centro_id)
f = open('human/human_centroid.txt', 'a')
for cen_id in centro_id:
    f.write(str(cen_id[0])+','+str(cen_id[1])+','+str(cen_id[2])+'\n')
f.close()

'''
# matching relationship generate

for line in open('mouse/mouse_centroid.txt'):
    mouse = (float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2]))
    euclidean_distance = []
    for lines in open('human/human_centroid.txt'):
        human = (float(lines.split(",")[0]), float(lines.split(",")[1]), float(lines.split(",")[2]))
        euclidean_distance.append(distance.euclidean(mouse, human))
    ind = euclidean_distance.index(min(euclidean_distance))
    euclidean_distance.clear()
    print(ind)
'''
'''
match_results = []
f = open('match_result.txt', 'a')
for match in open('matching-clustering.txt'):
    mouse_cluster_dir = match.split(',')[0].replace('\n', '')
    human_cluster_dir = match.split(',')[1].replace('\n', '')
    print(mouse_cluster_dir+','+human_cluster_dir)
    for line in open('mouse/'+str(mouse_cluster_dir)+'.txt'):
        mouse_value = (float(line.split(",")[0]),  float(line.split(",")[2]))
        check = []
        human_storage = []
        for lines in open('human/' + str(human_cluster_dir) + '.txt'):
            human_value = (float(lines.split(",")[0]),  float(lines.split(",")[2]))
            human_storage.append(human_value)
            check.append(distance.euclidean(mouse_value, human_value))
        human_storage[check.index(min(check))]
        f.write(str(float(line.split(",")[0])) + ',' + str(human_storage[check.index(min(check))][1]) + '\n')
        f.write(str(float(line.split(",")[2])) + ',' + str(human_storage[check.index(min(check))][0]) + '\n')
f.close()

'''
'''
# matching evaluation
result = {}
result_test = []
for line in open('match_result.txt'):
    result[str(float(line.split(",")[0]))] = str(float(line.split(",")[1].replace('\n', '')))
    result_test.append([str(line.split(",")[0]) + str(line.split(",")[1].replace('\n', ''))])
reference = {}
reference_test = []
for lines in open('match_reference.txt'):
    reference[str(float(lines.split(",")[0]))] = str(float(lines.split(",")[1].replace('\n', '')))
    reference_test.append([str(lines.split(",")[0]) + str(lines.split(",")[1].replace('\n', ''))])
a = result.keys() & reference.keys()
b = result.items() & reference.items()

print(len(a))
retA = [i for i in reference.values() if i in result.values()]
print(b)
print(len(retA))
i = 0
for a_s in a:
    print(str(i)+": "+reference[a_s] is result[a_s])
    i = i + 1
'''