l = list()
cnt =0
f = open("train2id.txt","r")
f2 = open("train2id.txt", "w")
f.readline()

for line in f:
    now = list(map(int,line.split(' ')))

    if now not in l:
        cnt +=1
        l.append(now)

print(cnt)

f2.write("%d\n" %len(l))

for a,b,c in l:
    f2.write("%d %d %d\n" %(a,b,c))

print(len(l))

