from scipy.spatial import distance
import logging

logger = logging.getLogger("Instance_matching")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

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
                mouse_value = (float(s_mouse), float(p_mouse),float(o_mouse))
                distance_check = []
                human_storage = []

                for key_human, value_human in human_filter.items():
                    s_human, p_human, o_human = key_human.split(',')
                    human_value = (float(s_human), float(p_human), float(o_human))
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

    f = open('3d_comp_match_result.txt', 'a')
    for ite in match_results:
        f.write(str(ite) + '\n')
    f.close()

ontology_matching()