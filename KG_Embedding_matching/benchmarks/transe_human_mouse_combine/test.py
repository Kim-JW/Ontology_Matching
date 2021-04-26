d = {}

d[0] = [1,2,3]
d[1] = [2]
d[2] = [4,5,6,7]

print(d.values())

if 1 in d.values():
    print("ok")
else:
    print("No")

for i in d.values():
    if 1 in i:
        print("Ok")