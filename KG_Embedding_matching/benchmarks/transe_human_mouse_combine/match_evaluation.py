import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import logging

logger = logging.getLogger("Instance_matching")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

def reference_data_get():
    f = open('benchmarks/transe_matching_instance/reference.txt', 'a')
    lis = []
    for line in open("./benchmarks/transe_matching_instance/transe_reference.txt"):
        if line != '\n':
            line = line.replace("\n", "")
            lis.append(line)
    lis = np.array(lis).reshape(1516, 2)
    print(lis)
    for lines in lis:
        f.write(str(lines)+'\n')
    f.close()

    f = open('match_reference.txt', 'a')
    mouse = []
    for line in open('transe_mouse_reference_triple2vector.txt'):
        line = line.replace("\n", "")
        mouse.append(line)

    human = []
    for line in open('transe_human_reference_triple2vector.txt'):
        line = line.replace("\n", "")
        human.append(line)

    for i in range(len(human)):
        f.write(str(mouse[i])+','+str(human[i])+'\n')
    f.close()


# Clustering Vectors using DBScan
def Clustering_Vector():
    mouse = []
    for line in open('transe_mouse_Triple2Vector.txt'):
        mouse.append([float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])])

    human = []
    for lines in open('transe_human_Triple2Vector.txt'):
        human.append([float(lines.split(",")[0]), float(lines.split(",")[1]), float(lines.split(",")[2])])

    mouse = np.array(mouse)
    human = np.array(human)

    mouse_model = DBSCAN(eps=0.03, min_samples=1)
    mouse_model.fit_predict(mouse)

    human_model = DBSCAN(eps=0.03, min_samples=1)
    human_model.fit_predict(human)

    # print("number of mouse cluster found: {}".format(len(set(mouse_model.labels_))))

    logger.info("number of mouse cluster found : {}".format(len(set(mouse_model.labels_))))
    logger.info('cluster for each mouse point: {}'.format(mouse_model.labels_))

    #print('cluster for each mouse point: ', mouse_model.labels_)

    logger.info("number of mouse cluster found : {}".format(len(set(human_model.labels_))))
    logger.info('cluster for each mouse point: {}'.format(human_model.labels_))

    #print("number of human cluster found: {}".format(len(set(human_model.labels_))))
    #print('cluster for each human point: ', human_model.labels_)


    # Classify Data

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

def generate_centroid():
    centro_id = []
    for i in range(11):
        subject = []
        predicate = []
        object = []
        for line in open('mouse/' + str(i) + '.txt'):
            subject.append(float(line.split(",")[0]))
            predicate.append(float(line.split(",")[1]))
            object.append(float(line.split(",")[2]))
        centro_id.append([np.mean(subject), np.mean(predicate), np.mean(object)])
        subject.clear()
        predicate.clear()
        object.clear()


    # print(centro_id)
    logger.info("centro: {}".format(centro_id))

    f = open('mouse/mouse_centroid.txt', 'a')
    for cen_id in centro_id:
        f.write(str(cen_id[0])+','+str(cen_id[1])+','+str(cen_id[2])+'\n')
    f.close()


# matching relationship generate
def pairing_clustering():
    for line in open('mouse/mouse_centroid.txt'):
        mouse = (float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2]))
        euclidean_distance = []
        for lines in open('human/human_centroid.txt'):
            human = (float(lines.split(",")[0]), float(lines.split(",")[1]), float(lines.split(",")[2]))
            euclidean_distance.append(distance.euclidean(mouse, human))
        ind = euclidean_distance.index(min(euclidean_distance))
        euclidean_distance.clear()
        print(ind)


def ontology_matching_humanbase():
    human = []
    for line_human in open('human.txt'):
        human.append(str(line_human.replace('\n', '')))
    mouse = []
    for line_mouse in open('mouse.txt'):
        mouse.append(str(line_mouse.replace('\n', '')))

    human_vector = []
    for line_human in open('transe_human_triple2vector.txt'):
        human_vector.append(str(line_human.replace('\n', '')))
    mouse_vector = []
    for line_mouse in open('transe_mouse_triple2vector.txt'):
        mouse_vector.append(str(line_mouse.replace('\n', '')))

    human_dic = dict(zip(human_vector, human))
    mouse_dic = dict(zip(mouse_vector, mouse))

    print(len(human_dic))
    print(len(mouse_dic))

    match_results = set()
    for match in open('matching_cluster.txt'):
        mouse_cluster_dir = match.split(',')[0].replace('\n', '')
        human_cluster_dir = match.split(',')[1].replace('\n', '')

        print(mouse_cluster_dir + ',' + human_cluster_dir)

        mouse_filter = {}
        human_filter = {}

        for line in open('mouse/' + str(mouse_cluster_dir) + '.txt'):
            line = line.replace('\n', '')
            s_mouse, p_mouse, o_mouse = mouse_dic[line].split('@')
            if 'http://mouse.owl#' in s_mouse and 'http://mouse.owl#' in o_mouse:
                mouse_filter[line] = 1
            elif 'http://mouse.owl#' in s_mouse:
                mouse_filter[line] = 2
            elif 'http://mouse.owl#' in o_mouse:
                mouse_filter[line] = 3

        for line in open('human/' + str(human_cluster_dir) + '.txt'):
            line = line.replace('\n', '')
            s_human, p_human, o_human = human_dic[line].split('@')
            if 'http://human.owl#' in s_human and 'http://human.owl#' in o_human:
                human_filter[line] = 1
            elif 'http://human.owl#' in s_human:
                human_filter[line] = 2
            elif 'http://human.owl#' in o_human:
                human_filter[line] = 3

        logger.info("number of mouse_filter : {}".format(len(mouse_filter)))
        logger.info("number of human_filter : {}".format(len(human_filter)))

        # print(len(mouse_filter))
        # print(len(human_filter))

        while len(mouse_filter) != 0 and len(human_filter) != 0:
            for key_human, value_human in human_filter.items():
                s_human, p_human, o_human = key_human.split(',')
                human_value = (float(s_human), float(o_human))
                distance_check = []
                mouse_storage = []

                for key_mouse, value_mouse in mouse_filter.items():
                    s_mouse, p_mouse, o_mouse = key_mouse.split(',')
                    mouse_value = (float(s_mouse), float(o_mouse))
                    mouse_storage.append(key_mouse)
                    distance_check.append(distance.euclidean(mouse_value, human_value))

                result = mouse_storage[distance_check.index(min(distance_check))]
                distance_check.clear()

                if mouse_filter[result] == 1:
                    if value_human == 1:
                        match_results.add(str(s_human) + ',' + str(result.split(',')[0]))
                        match_results.add(str(o_human) + ',' + str(result.split(',')[2]))
                        match_results.add(str(s_human) + ',' + str(result.split(',')[2]))
                        match_results.add(str(o_human) + ',' + str(result.split(',')[0]))
                    if value_human == 2:
                        match_results.add(str(s_human) + ',' + str(result.split(',')[0]))
                        match_results.add(str(s_human) + ',' + str(result.split(',')[2]))
                    if value_human == 3:
                        match_results.add(str(o_human) + ',' + str(result.split(',')[0]))
                        match_results.add(str(o_human) + ',' + str(result.split(',')[2]))

                if mouse_filter[result] == 2:
                    if value_human == 1:
                        match_results.add(str(s_human) + ',' + str(result.split(',')[0]))
                        match_results.add(str(o_human) + ',' + str(result.split(',')[0]))
                    if value_human == 2:
                        match_results.add(str(s_human) + ',' + str(result.split(',')[0]))
                    if value_human == 3:
                        match_results.add(str(o_human) + ',' + str(result.split(',')[0]))

                if mouse_filter[result] == 3:
                    if value_human == 1:
                        match_results.add(str(o_human) + ',' + str(result.split(',')[2]))
                        match_results.add(str(s_human) + ',' + str(result.split(',')[2]))
                    if value_human == 2:
                        match_results.add(str(s_human) + ',' + str(result.split(',')[2]))
                    if value_human == 3:
                        match_results.add(str(o_human) + ',' + str(result.split(',')[2]))

            mouse_filter.clear()
            human_filter.clear()

    f = open('match_result_humanbase.txt', 'a')
    for ite in match_results:
        f.write(str(ite) + '\n')
    f.close()

def ontology_matching():
    human = []
    for line_human in open('human.txt'):
        human.append(str(line_human.replace('\n', '')))
    mouse = []
    for line_mouse in open('mouse.txt'):
        mouse.append(str(line_mouse.replace('\n', '')))

    human_vector = []
    for line_human in open('transe_human_triple2vector.txt'):
        human_vector.append(str(line_human.replace('\n', '')))
    mouse_vector = []
    for line_mouse in open('transe_mouse_triple2vector.txt'):
        mouse_vector.append(str(line_mouse.replace('\n', '')))

    human_dic = dict(zip(human_vector, human))
    mouse_dic = dict(zip(mouse_vector, mouse))

    print(len(human_dic))
    print(len(mouse_dic))

    match_results = set()
    for match in open('matching_cluster.txt'):
        mouse_cluster_dir = match.split(',')[0].replace('\n', '')
        human_cluster_dir = match.split(',')[1].replace('\n', '')

        print(mouse_cluster_dir + ',' + human_cluster_dir)

        mouse_filter = {}
        human_filter = {}

        for line in open('mouse/' + str(mouse_cluster_dir) + '.txt'):
            line = line.replace('\n', '')
            s_mouse, p_mouse, o_mouse = mouse_dic[line].split('@')
            if 'http://mouse.owl#' in s_mouse and 'http://mouse.owl#' in o_mouse:
                mouse_filter[line] = 1
            elif 'http://mouse.owl#' in s_mouse:
                mouse_filter[line] = 2
            elif 'http://mouse.owl#' in o_mouse:
                mouse_filter[line] = 3

        for line in open('human/' + str(human_cluster_dir) + '.txt'):
            line = line.replace('\n', '')
            s_human, p_human, o_human = human_dic[line].split('@')
            if 'http://human.owl#' in s_human and 'http://human.owl#' in o_human:
                human_filter[line] = 1
            elif 'http://human.owl#' in s_human:
                human_filter[line] = 2
            elif 'http://human.owl#' in o_human:
                human_filter[line] = 3

        logger.info("number of mouse_filter : {}".len(format(mouse_filter)))
        logger.info("number of human_filter : {}".len(format(human_filter)))

        #print(len(mouse_filter))
        #print(len(human_filter))

        while len(mouse_filter) != 0 and len(human_filter) != 0:
            for key_mouse, value_mouse in mouse_filter.items():
                s_mouse, p_mouse, o_mouse = key_mouse.split(',')
                mouse_value = (float(s_mouse), float(o_mouse))
                distance_check = []
                human_storage = []

                for key_human, value_human in human_filter.items():
                    s_human, p_human, o_human = key_human.split(',')
                    human_value = (float(s_human), float(o_human))
                    human_storage.append(key_human)
                    distance_check.append(distance.euclidean(mouse_value, human_value))

                result = human_storage[distance_check.index(min(distance_check))]
                distance_check.clear()

                if human_filter[result] == 1:
                    if value_mouse == 1:
                        match_results.add(str(s_mouse)+','+str(result.split(',')[0]))
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[2]))
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[2]))
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[0]))
                    if value_mouse == 2:
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[0]))
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[2]))
                    if value_mouse == 3:
                        match_results.add(str(o_mouse)+','+str(result.split(',')[0]))
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[2]))

                if human_filter[result] == 2:
                    if value_mouse == 1:
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[0]))
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[0]))
                    if value_mouse == 2:
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[0]))
                    if value_mouse == 3:
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[0]))

                if human_filter[result] == 3:
                    if value_mouse == 1:
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[2]))
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[2]))
                    if value_mouse == 2:
                        match_results.add(str(s_mouse) + ',' + str(result.split(',')[2]))
                    if value_mouse == 3:
                        match_results.add(str(o_mouse) + ',' + str(result.split(',')[2]))

            mouse_filter.clear()
            human_filter.clear()

    f = open('match_result.txt', 'a')
    for ite in match_results:
        f.write(str(ite) + '\n')
    f.close()


# matching evaluation
def matching_evaluation():
    result = {}
    result_test = set()
    # checkdd = set()
    for line in open('match_result.txt'):
        result[str(float(line.split(",")[0]))] = str(float(line.split(",")[1]))
        result_test.add(str(line.split(",")[0]) + str(line.split(",")[1]))
        # checkdd.add(float(line.split(",")[1]))
    reference = {}
    reference_test = set()

    for lines in open('match_reference.txt'):
        reference[str(float(lines.split(",")[0]))] = str(float(lines.split(",")[1]))
        reference_test.add(str(lines.split(",")[0]) + str(lines.split(",")[1]))


    a = result.keys() & reference.keys()
    b = result.items() & reference.items()

    print(len(a))
    print(b)
    retA = [i for i in result.values() if i in reference.values()]
    print(len(retA))

def main():
    logger.info("Ontology_Mathcing_Begin")

    #reference_data_get()

    #Clustering_Vector()

    #generate_centroid()

    #pairing_clustering()

    #ontology_matching()

    ontology_matching_humanbase()

    #matching_evaluation()

if __name__ == '__main__':
    main()
