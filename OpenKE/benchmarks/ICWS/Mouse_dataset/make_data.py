E = list()
P = list()
cnt = 0

for line in open('mouse.txt', 'r'):

    cnt +=1
    ss = line.split('@')[0].strip()
    pp = line.split('@')[1].strip()
    oo = line.split('@')[2].strip()

    if ss not in E:
        E.append(ss)
    if pp not in P:
        P.append(pp)
    if oo not in E:
        E.append(oo)

#train2id

f2 = open("Mouse_train2id.txt", "w")
f2.write("%d\n" %cnt)

for line in open("mouse.txt", "r"):
    ss = line.split('@')[0].strip()
    pp = line.split('@')[1].strip()
    oo = line.split('@')[2].strip()

    f2.write("%d\t%d\t%d\n" %(E.index(ss), E.index(oo), P.index(pp)))

f2.close()


# entity
f = open("Mouse_entity2id.txt", "w")
f.write("%d\n" %len(E))

for i in range(len(E)):
    f.write("%s\t%d\n" %(E[i], i))
f.close()


# relation
f = open("Mouse_relation2id.txt", "w")
f.write("%d\n" %len(P))

for i in range(len(P)):
    f.write("%s\t%d\n" %(P[i], i))
f.close()

