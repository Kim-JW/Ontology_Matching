def e():

    d = {}
    ref = []

    reverse_d = {}

    cnt = 0

    for line in open("3d_comparision_match_result.txt",'r'):
        m = float(line.split(",")[0]) # Mouse
        h = float(line.split(",")[1].replace("\n","")) # Human

        if m in d:
            d[m].append(h)
        else:
            d[m] = [h]

        if h in reverse_d:
            reverse_d[h].append(m)
        else:
            reverse_d[h] = [m]

    # print(d)

    tt = 0

    for line in open("match_reference.txt"):
        r1 = float(line.split(",")[0]) # Mouse
        r2 = float(line.split(",")[1].replace("\n","")) # Human

        if r1 in d:
            tt +=1
            for i in d[r1]:
                if i == r2:
                    cnt+=1
        
        if r2 in reverse_d:
            for i in reverse_d[r2]:
                if i == r1:
                    cnt +=1

            #if r2 in d[r1]:
            #     cnt+=1

    print(tt)
    print("Number of mathcing is " , cnt)

    mcnt =0

    mh = set()

    for line in open("transe_mouse_triple2vector.txt","r"):
        r1 = float(line.split(",")[0])
        r2 = float(line.split(",")[1].replace("\n", ""))

        mh.add(r1)

    for i in ref:
        if i in mh:
            mcnt +=1

    print(mcnt)

def matching_evaluation():
    result = {}
    result_test = set()
    # checkdd = set()
    for line in open('3d_comparision_match_result.txt'):
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

matching_evaluation()

e()