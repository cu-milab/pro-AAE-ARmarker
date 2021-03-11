import math
import numpy as np

def calc_cos(list1, list2):
    #print("bef",list2)
    #cos_sim.pyを動かすとき，リストの最後にファイル名をつけている(str)ので，リストの最後は無視するようにする
    list1 = list1[:-1]
    list2 = list2[:-1]
    #print("af",list2)
    list1 = np.array(list1) # list ではなく np.array を使う list 2 が候補の潜在変数
    list2 = np.array(list2)
    #print("1",list1)
    #print("2",list2)
    sum1 = sum(list1**2) # ループを使う必要がない（速い）
    sum2 = sum(list2**2)
    length1 = math.sqrt(sum1)
    length2 = math.sqrt(sum2)
    inner_product = sum(list1*list2) # ループを使う必要がない（速い）
    if (length1 != 0 and length2 != 0):
        cos = inner_product/(length1*length2)
    else:
        cos = 0
    return cos