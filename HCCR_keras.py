# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
os.chdir("G:/DataSet/detail")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
from keras import backend as K
K.image_data_format()
K.set_image_data_format('channels_first')
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model, model_from_json, model_from_yaml
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import gc
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import cv2

# load data
f = open("./char_dict", "rb")
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

char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"

img_size = 96
X = np.zeros((img_size, img_size), dtype="uint8")
X = np.expand_dims(X, axis=0)
Y = np.zeros(1, dtype="uint8")

dim = "test/" # train/
flag = -1
for folder_name in char_df.folder:
    if char_df[char_df.folder == folder_name].word.values[0] in char_set:
        os.chdir("G:/DataSet/detail/" + dim + folder_name)
        print("into folder: " + folder_name + "; time: " + str(pd.Timestamp.now())[0:19])
        flag += 1
        images_list = os.listdir()
        
        Y_tmp = np.tile(flag, len(images_list))
        Y = np.concatenate((Y, Y_tmp))
        
        X_tmp = np.zeros((img_size, img_size), dtype="uint8")
        X_tmp = np.expand_dims(X_tmp, axis=0)
            
        for file_name in images_list:
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

trainSet_140 = {"target":X, "label":Y}
f = open("G:/DataSet/detail/trainSet_140.txt", "wb")
pickle.dump(trainSet_140, f); f.close()

testSet_140 = {"target":X, "label":Y}
f = open("G:/DataSet/detail/testSet_140.txt", "wb")
pickle.dump(testSet_140, f); f.close()

# model
f = open("G:/DataSet/detail/trainSet_140.txt", "rb")
trainSet_140 = pickle.load(f); f.close()
x_train = trainSet_140["target"]; y_train = trainSet_140["label"]
x_train, y_train = shuffle(x_train, y_train, random_state=0)

N, H, W = x_train.shape
x_train = x_train.reshape(N, 1, H, W)
val_max = 255
x_train = (x_train/val_max).astype("float32")

#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
#itemfreq(y_valid)
y_train_ot = to_categorical(y_train)
#y_valid_ot = to_categorical(y_valid)

inpt = Input(shape=(1, 96, 96))
# 1
x = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(inpt)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (96, 48, 48)
# 2
x = Conv2D(filters=128, kernel_size=(3,1), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=128, kernel_size=(1,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (128, 24, 24)
# 3
x = Conv2D(filters=160, kernel_size=(3,1), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=160, kernel_size=(1,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (160, 12, 12)
# 4
x = Conv2D(filters=256, kernel_size=(3,1), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=256, kernel_size=(1,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 5
x = Conv2D(filters=256, kernel_size=(3,1), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=256, kernel_size=(1,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (256, 6, 6)
# 6
x = Conv2D(filters=384, kernel_size=(3,1), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=384, kernel_size=(1,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 7
x = Conv2D(filters=384, kernel_size=(3,1), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=384, kernel_size=(1,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (384, 3, 3)
# 8
x = Flatten()(x)
x = Dense(units=1024, activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = Dropout(rate=0.5)(x)
# 9
x = Dense(units=140, activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = Activation("softmax")(x)
#x.shape

model = Model(inputs=inpt, outputs=x)
#model.summary()
#plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1, momentum=0.9, decay=0.1), metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0), metrics=["accuracy"])
bs = 256
model.fit(x_train, y_train_ot, batch_size=bs, epochs=5, verbose=1) 
# validation_data=(x_valid, y_valid_ot); validation_split=0.2

## pred
f = open("G:/DataSet/detail/testSet_140.txt", "rb")
testSet_140 = pickle.load(f); f.close()
x_test = testSet_140["target"]; y_test = testSet_140["label"]
x_test, y_test = shuffle(x_test, y_test, random_state=0)

N, H, W = x_test.shape
x_test = x_test.reshape(N, 1, H, W)
val_max = 255
x_test = (x_test/val_max).astype("float32")
y_test_ot = to_categorical(y_test)

loss_train, accu_train = model.evaluate(x_train, y_train_ot, verbose=1)
loss_valid, accu_valid = model.evaluate(x_test, y_test_ot, verbose=1)

output = model.predict(x_test, batch_size=bs, verbose=1)
y_pred = np.argmax(output, axis=1)
sum(y_pred == y_test) / len(y_test) # 0.9249
pd.crosstab(y_test, y_pred, margins=True)

# 模型和权重
model.save('my_model.h5')
model = load_model('my_model.h5')
# 模型
json_string = model.to_json()
model = model_from_json(json_string)
yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
# 权重
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5', by_name=True)
