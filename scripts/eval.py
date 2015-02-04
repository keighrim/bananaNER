
with open("../dataset/dev.gold") as gold, open("../result/result.txt") as res:
    cor = 0
    inc = 0
    tmp = next(gold)
    tmp_r = next(res)
    while (tmp != None):
        # print tmp, tmp_r
        if len(tmp) <= 1:
            pass
        else:
            g = tmp.split("\t")[3]
            r = tmp_r.split("\t")[3]
            print g, r
            if g == r:
                if g == "O":
                    cor += 1
            else:
                inc +=1
        try:
            tmp = next(gold)
            tmp_r = next(res)
        except StopIteration:
            tmp = None
            print "THE END"

    print cor
    print inc
