f = open('./benchmarks/FB15K237/Hash_Triple2HashValues_100.txt', 'a')
for line in open("./benchmarks/FB15K237/train.txt"):
    line = line.replace("\n", "")
    # rel_keys.append(line.split("\t")[0])
    # print(ent_dic.get(line.split("#")[0]))
    # print(rel_dic.get(line.split("#")[1]))
    # print(ent_dic.get(line.split("#")[2]))
    f.write(str(line.split("#")[0].__hash__()) + "," + str(line.split("#")[1].__hash__()) + "," + str(line.split("#")[2].__hash__()) + "\n")
f.close()
