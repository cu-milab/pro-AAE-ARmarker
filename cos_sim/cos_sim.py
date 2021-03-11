import np_cos_sim
import numpy as np
import glob
import math
from sklearn.metrics import mean_absolute_error


#pose_test = open("AAE_test/.txt", "r")
#評価用データ（推定対象）（潜在変数）
test_path = "/home/milab/1st_AAE/AAE_py/pro-depth-image-interpolation-using-ae/augument_autoencoder/SSD_hyouka2/SSD_0_20/"

#print(test_path)
#データベース（潜在変数）
pose_path = "/home/milab/1st_AAE/AAE_py/pro-depth-image-interpolation-using-ae/augument_autoencoder/data_base/SSD_AR0_20_last/"

pose_list_folder = []   #データベースのリスト
test_list_folder = []   #評価用のリスト
#データベースの潜在変数の読みこみ，リスト化
folder = sorted(glob.glob(pose_path + "*.txt"))
for num ,file_path in enumerate(folder):
    with open(file_path) as f:
        a = f.read().splitlines()
        b = [float(c) for c in a]
        append_path = file_path.replace(pose_path, "")
        b.append(append_path)
        pose_list_folder.append(b)
        

folder_test = sorted(glob.glob(test_path + "*.txt"))
for num, file_path in enumerate(folder_test):
    with open(file_path) as f:
        a = f.read().splitlines()
        b = [float(c) for c in a]
        append_path = file_path.replace(test_path, "")
        b.append(append_path)
        test_list_folder.append(b)


r_test = []
p_test = []
y_test = []

r_pose = []
p_pose = []
y_pose = []
for i in range(len(test_list_folder)):
    cos_list = []
    for n in range(len(pose_list_folder)):
        #print("test",test_list_folder[i])
        #print("pose",pose_list_folder[n])
        cos0 = np_cos_sim.cos_similarity(test_list_folder[i] ,pose_list_folder[n])
        cos_list.append(cos0)
    cos_list = np.array(cos_list)
    max2 = cos_list.argmax()    #リストのなかの何番目
    #test_name 
    aa = folder_test[i]
    aa = aa.replace(test_path, "")
    aa = aa.replace(".txt","")
    print("正解姿勢は",aa)
    #pose_name
    folder[max2] = folder[max2].replace(pose_path, "")
    folder[max2] = folder[max2].replace(".txt","")
    bb = folder[max2]
    print("推定姿勢は", bb)

    print("###########################################################################")

    #数字のみを取り出すから．ｒ，ｐ，ｙの位置を探す
    aa_r_pos = aa.find("r")
    aa_p_pos = aa.find("p")
    aa_y_pos = aa.find("y")

    bb_r_pos = bb.find("r")
    bb_p_pos = bb.find("p")
    bb_y_pos = bb.find("y")

    aa_r = aa[aa_r_pos+1:aa_p_pos] #pの数字
    aa_p = aa[aa_p_pos+1:aa_y_pos] #pの数字
    aa_y = aa[aa_y_pos+1:] #pの数字

    bb_r = bb[bb_r_pos+1:bb_p_pos] #pの数字
    bb_p = bb[bb_p_pos+1:bb_y_pos] #pの数字
    bb_y = bb[bb_y_pos+1:] #pの数字

    r_test.append(aa_r)
    p_test.append(aa_p)
    y_test.append(aa_y)

    r_pose.append(bb_r)
    p_pose.append(bb_p)
    y_pose.append(bb_y)


r_pose = [float(v) for v in r_pose]
p_pose = [float(v) for v in p_pose]
y_pose = [float(v) for v in y_pose]
r_test = [float(v) for v in r_test]
p_test = [float(v) for v in p_test]
y_test = [float(v) for v in y_test]


r_square_list = []
p_square_list = []
y_square_list = []
#rmseの計算
for i in range(len(r_test)): #86回
    r_square = (r_test[i] - r_pose[i])**2
    p_square = (p_test[i] - p_pose[i])**2
    y_square = (y_test[i] - y_pose[i])**2
    r_square_list.append(r_square)
    p_square_list.append(p_square)
    y_square_list.append(y_square)

r_mean = np.mean(r_square_list)
p_mean = np.mean(p_square_list)
y_mean = np.mean(y_square_list)

r_root = math.sqrt(r_mean)
p_root = math.sqrt(p_mean)
y_root = math.sqrt(y_mean)

#MAE

print("MAE_r",mean_absolute_error(r_pose, r_test))
print("MAE_p",mean_absolute_error(p_pose, p_test))
print("MAE_y",mean_absolute_error(y_pose, y_test))

#MSE
print("r",r_mean)
print("p",p_mean)
print("y",y_mean)

#RMSE
print("r_r",r_root)
print("p_r",p_root)
print("y_r",y_root)
    # ここからRMSEの計算箇所
    #printしたtextからの値の抽出
    #リストに入れる
    #引き算，２乗，和，平均(/86)


    #姿勢を表示する



# chemistry_list = []
# cos_list = [] # cos だけを記録するリスト
# n = len(list_folder)
# for i in range(n):    #len(list_folder) is 27036
#     for j in range(128): #128
#         #潜在変数をひとつずつ比較する
#         print(list_folder[i][j])
#         cos0 = module_cos.calc_cos(test_list[j], list_folder[i][j])
#         #temp = tag[i]+"と"+tag[j]+"："+str(cos0)
#         #chemistry_list.append(temp)
#         cos_list.append(cos0)
#         #print(temp)
# cos_list = np.array(cos_list)
# max2 = cos_list.argmax() # 最大値は何番目の要素かを求める
# print("推定された姿勢は", chemistry_list[max2])
