f = open("test.txt",'r')

f.readline()

t = tuple()

for line in f:
    a,b,c = line.split(' ')
    t = t + (a,b,c)

for i in range(len(t)):
    print(t[i])
