from tqdm import tqdm
from time import time

cnt =0

h = []
m = []

for line in open("transe_human_triple2vector.txt"):
    x,y,z = float(line.split(",")[0]), float(line.split(",")[1]), float(line.split(",")[2])
    h.append([x,y,z])


for lines in open("transe_mouse_triple2vector.txt"):
    mx,my,mz = float(lines.split(",")[0]),float(lines.split(",")[1]),float(lines.split(",")[2])
    m.append([mx,my,mz])


for i in tqdm(range(len(h))):

    for j in range(len(m)):
        if h[i] == m[j]:
            cnt +=1
