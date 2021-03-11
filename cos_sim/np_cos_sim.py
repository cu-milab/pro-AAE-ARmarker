import numpy as np

def cos_similarity(list1, list2):
    #print("bef",list2)
    #cos_sim.pyを動かすとき，リストの最後にファイル名をつけている(str)ので，リストの最後は無視するようにする
    list1 = list1[:-1]
    list2 = list2[:-1]
    #print("af",list2)
    list1 = np.array(list1) # list ではなく np.array を使う list 2 が候補の潜在変数
    list2 = np.array(list2)
    cos = np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))
    return cos