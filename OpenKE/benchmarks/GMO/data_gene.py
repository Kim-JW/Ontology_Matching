E = list()
P = list()
cnt = 0

f2 = open("gbobcodmo2RDF_Processing.txt",'w')

for line in open('gbobcodmo2RDF.txt','r'):

    length = line.split('\t')

    if len(length) == 3:

        cnt +=1
        ss = line.split('\t')[0]
        pp = line.split('\t')[1]
        oo = line.split('\t')[2]

        f2.write(ss + '\t' + pp + '\t' + oo + '\n')

        if ss not in E:
            E.append(ss)
        if pp not in P:
            P.append(pp)
        if oo not in E:
            E.append(oo)

#train2id

'''for line in open("gbobcodmo2RDF.txt","r"):
    ss = line.split('\t')[0]
    pp = line.split('\t')[1]
    oo = line.split('\t')[2]

    if ss in E and oo in E and pp in P:

        print("%d\t%d\t%d\n" %(E.index(ss), E.index(oo), P.index(pp)))'''
