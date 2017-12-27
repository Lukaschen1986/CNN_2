# -*- coding: utf-8 -*-
# http://pdfs.semanticscholar.org/0752/8274309b357651919c59bea8fdafa1116277.pdf
# http://blog.csdn.net/ssbqrm/article/details/73227437
from __future__ import absolute_import, division, print_function
import os
os.chdir("G:/DataSet/detail")
os.getcwd()
import gc
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

f = open("./HWDB1/char_dict", "rb")
char_dict = pickle.load(f); f.close()

words = list(char_dict.keys())
folders = list(char_dict.values())
folders_res = []
for folder in folders:
    if len(str(folder)) == 1:
        folder_new = "0000" + str(folder)
    elif len(str(folder)) == 2:
        folder_new = "000" + str(folder)
    elif len(str(folder)) == 3:
        folder_new = "00" + str(folder)
    else:
        folder_new = "0" + str(folder)
    folders_res.append(folder_new)
len(folders_res)

char_df = pd.DataFrame({"word":words, "folder":folders_res}, columns=["word","folder"])
char_df = char_df.sort_values(by="folder", ascending=True)

#char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

img_size = 96
X = np.zeros((img_size, img_size), dtype="uint8")
X = np.expand_dims(X, axis=0)
Y = np.zeros(1, dtype="uint8")

flag = -1
for folder_name in char_df.folder:
#for folder_name in ("00001","00002"):    
#    folder_name = "00002"
    os.chdir("G:/DataSet/HWDB1/test/" + folder_name)
    print("into folder: " + folder_name + "; time: " + str(pd.Timestamp.now())[0:19])
    flag += 1
    images_list = os.listdir()
    
    Y_tmp = np.tile(flag, len(images_list))
    Y = np.concatenate((Y, Y_tmp))
    
    X_tmp = np.zeros((img_size, img_size), dtype="uint8")
    X_tmp = np.expand_dims(X_tmp, axis=0)
        
    for file_name in images_list:
        # file_name = "101330.png"
        pic = Image.open(file_name, mode="r")
        pic_resize = pic.resize((img_size, img_size), Image.BICUBIC)
            
        pic_array = image.img_to_array(pic_resize, "channels_last").astype("uint8") # img_to_array
        pic_grey = cv2.cvtColor(pic_array, code=cv2.COLOR_BGR2GRAY) # 灰度化
        pic_threshold = cv2.threshold(pic_grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1] # 二值化 与 黑白翻转
#        Image.fromarray(pic_threshold)
        pic_new = np.expand_dims(pic_threshold, axis=0) # 变为一通道
        X_tmp = np.concatenate((X_tmp, pic_new), axis=0)
        
    X_tmp = X_tmp[1:]
    X = np.concatenate((X, X_tmp), axis=0)

X = X[1:] # 删除第一个0数据
Y = Y[1:]
testSet = {"target":X, "label":Y}
f = open("G:/DataSet/HWDB1/testSet.txt", "wb")
pickle.dump(testSet, f); f.close()

f = open("G:/DataSet/HWDB1/trainSet_3.txt", "rb")
trainSet_1 = pickle.load(f)
trainSet_2 = pickle.load(f)
trainSet_3 = pickle.load(f)
f.close()
