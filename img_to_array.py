import cv2
import os
import numpy as np

kelas_0 = "dataset/0 Sangat Kotor"
kelas_1 = "dataset/1 Kotor"
kelas_2 = "dataset/2 Sedikit Bersih"
kelas_3 = "dataset/3 Bersih"


def list_data(folder):
    data_temp=[]
    for filename in os.listdir(folder):
        data_temp.append(folder+'/'+filename)
    data_temp=np.array(data_temp) 
    return data_temp

kelas0 = list_data(kelas_0)
kelas1 = list_data(kelas_1)
kelas2 = list_data(kelas_2)
kelas3 = list_data(kelas_3)

np.save("numpy/path_0.npy", kelas0)
np.save("numpy/path_1.npy", kelas1)
np.save("numpy/path_2.npy", kelas2)
np.save("numpy/path_3.npy", kelas3)

file_kelas_0 = np.load("numpy/path_0.npy")
file_kelas_1 = np.load("numpy/path_1.npy")
file_kelas_2 = np.load("numpy/path_2.npy")
file_kelas_3 = np.load("numpy/path_3.npy")
##
##print(file_kelas_0.shape)
##print(file_kelas_1.shape)
##print(file_kelas_2.shape)
##print(file_kelas_3.shape)

def get_corner_value(file):
    img_name = file
    img = cv2.imread(img_name)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    fast = cv2.FastFeatureDetector_create()
    kp_with_nonmax = fast.detect(gray,None)

    fast.setNonmaxSuppression(False)
    kp_without_nonmax = fast.detect(gray,None)

    img_with_nonmax = np.copy(rgb)
    img_without_nonmax = np.copy(rgb)
    img2 = cv2.drawKeypoints(rgb,kp_with_nonmax,img_with_nonmax,color=(255,0,0), flags=0)
    img3 = cv2.drawKeypoints(rgb,kp_without_nonmax,img_without_nonmax,color=(255,0,0), flags=0)
    
    data_with_nonmax = len(kp_with_nonmax)
    data_without_nonmax = len(kp_without_nonmax)
    return data_without_nonmax, data_with_nonmax

test=file_kelas_0[22]
get_corner_value(test)

def separate_dataset(data,tipe_kelas):
    temp_list = []
    for i in range(len(data)):
        test = data[i]
        nms, non_nms = get_corner_value(test)

        temp = [nms, non_nms, tipe_kelas]
        temp_list.append(temp)
    return temp_list

list_kelas_0 = separate_dataset(file_kelas_0, "Sangat Kotor")
list_kelas_1 = separate_dataset(file_kelas_1, "Kotor")
list_kelas_2 = separate_dataset(file_kelas_2, "Sedikit Bersih")
list_kelas_3 = separate_dataset(file_kelas_3, "Bersih")

np.save("numpy/kelas_0.npy", list_kelas_0)
np.save("numpy/kelas_1.npy", list_kelas_1)
np.save("numpy/kelas_2.npy", list_kelas_2)
np.save("numpy/kelas_3.npy", list_kelas_3)

kelas_0 = np.load("numpy/kelas_0.npy")
kelas_1 = np.load("numpy/kelas_1.npy")
kelas_2 = np.load("numpy/kelas_2.npy")
kelas_3 = np.load("numpy/kelas_3.npy")

print(kelas_0.shape)
print(kelas_1.shape)
print(kelas_2.shape)
print(kelas_3.shape)
