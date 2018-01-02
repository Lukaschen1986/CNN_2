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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
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

char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感订单己确认取消"

img_size = 96
X = np.zeros((img_size, img_size), dtype="uint8")
X = np.expand_dims(X, axis=0)
Y = np.zeros(1, dtype="uint8")
Z = []

dim = "test/" # train/
flag = -1
for char in char_set:
    # char = "的"
    if char in char_df.word.values:
        folder_name = char_df[char_df.word.values == char].folder.values[0]
        os.chdir("G:/DataSet/detail/" + dim + folder_name)
        flag += 1
        images_list = os.listdir()
        # save Y
        Y_tmp = np.tile(flag, len(images_list))
        Y = np.concatenate((Y, Y_tmp))
        # save Z
        Z.append(char)
        # save X
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
    else:
        print("the char isn't in folder")
        continue

#for folder_name in char_df.folder:
#    if char_df[char_df.folder == folder_name].word.values[0] in char_set:
#        os.chdir("G:/DataSet/detail/" + dim + folder_name)
#        print("into folder: " + folder_name + "; time: " + str(pd.Timestamp.now())[0:19])
#        flag += 1
#        images_list = os.listdir()
#        
#        Y_tmp = np.tile(flag, len(images_list))
#        Y = np.concatenate((Y, Y_tmp))
#        
#        X_tmp = np.zeros((img_size, img_size), dtype="uint8")
#        X_tmp = np.expand_dims(X_tmp, axis=0)
#            
#        for file_name in images_list:
#            pic = Image.open(file_name, mode="r")
#            pic_resize = pic.resize((img_size, img_size), Image.BICUBIC)
#                
#            pic_array = image.img_to_array(pic_resize, "channels_last").astype("uint8") # img_to_array
#            pic_grey = cv2.cvtColor(pic_array, code=cv2.COLOR_BGR2GRAY) # 灰度化
#            pic_threshold = cv2.threshold(pic_grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1] # 二值化 与 黑白翻转
#    #        Image.fromarray(pic_threshold)
#            pic_new = np.expand_dims(pic_threshold, axis=0) # 变为一通道
#            X_tmp = np.concatenate((X_tmp, pic_new), axis=0)
#            
#        X_tmp = X_tmp[1:]
#        X = np.concatenate((X, X_tmp), axis=0)

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

'''
c = 96; h = 48; w = 48; n = 128; k = 3
h*w*n*k*k*c / 1024 / 1024 # 无瓶颈层大小：243
speed = 1
d = (c*n*k*w) / (c*w+n*w) / speed
h*w*d*k*1*c / 1024 / 1024 # 有瓶颈层大小：26
h*w*n*1*k*d / 1024 / 1024 # 有瓶颈层大小：34
'''

def d(c, n, k, w, speed):
    val = (c*n*k*w) / (c*w+n*w) / speed
    val = int(val)
    return val

param = {"ck":3, "cs":1, 
         "mpk":3, "mps":2, 
         "apk":3, "aps":2, 
         "speed":4, "reg":0.01,
         "h0":96, "w0":96, "c0":1, 
         "n1":96, "n2":128, "n3":192,
         "n4":256, "n5":256, "n6":384,
         "n7":384, "n8":1024, "n9":140}

# Model_1
inpt = Input(shape=(param["c0"], param["h0"], param["w0"]))
# 1
x = Conv2D(filters=param["n1"], kernel_size=(param["k"],param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(inpt)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["k"],param["k"]), strides=param["ps"], padding="same")(x) 
x.shape # (96, 48, 48)
# 2
d2 = d(c=param["n1"], n=param["n2"], k=param["k"], w=int(x.shape[2]), speed=param["speed"])
x = Conv2D(filters=d2, kernel_size=(param["k"],1), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=param["n2"], kernel_size=(1,param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) 
x.shape # (128, 24, 24)
# 3
d3 = d(c=param["n2"], n=param["n3"], k=param["k"], w=int(x.shape[2]), speed=param["speed"])
x = Conv2D(filters=d3, kernel_size=(param["k"],1), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=param["n3"], kernel_size=(1,param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["k"],param["k"]), strides=param["ps"], padding="same")(x) 
x.shape # (160, 12, 12)
# 4
d4 = d(c=param["n3"], n=param["n4"], k=param["k"], w=int(x.shape[2]), speed=param["speed"])
x = Conv2D(filters=d4, kernel_size=(param["k"],1), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=param["n4"], kernel_size=(1,param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 5
d5 = d(c=param["n4"], n=param["n5"], k=param["k"], w=int(x.shape[2]), speed=param["speed"])
x = Conv2D(filters=d5, kernel_size=(param["k"],1), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=param["n5"], kernel_size=(1,param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["k"],param["k"]), strides=param["ps"], padding="same")(x) # (256, 6, 6)
# 6
d6 = d(c=param["n5"], n=param["n6"], k=param["k"], w=int(x.shape[2]), speed=param["speed"])
x = Conv2D(filters=d6, kernel_size=(param["k"],1), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=param["n6"], kernel_size=(1,param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 7
d7 = d(c=param["n6"], n=param["n7"], k=param["k"], w=int(x.shape[2]), speed=param["speed"])
x = Conv2D(filters=d7, kernel_size=(param["k"],1), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = Conv2D(filters=param["n7"], kernel_size=(1,param["k"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["k"],param["k"]), strides=param["ps"], padding="same")(x) # (384, 3, 3)
# 8
x = Flatten()(x)
x = Dense(units=param["n8"], activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = Dropout(rate=0.5)(x)
# 9
x = Dense(units=param["n9"], activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = Activation("softmax")(x)
#x.shape

# Model_2
inpt = Input(shape=(param["c0"], param["h0"], param["w0"]))
# 1
x = Conv2D(filters=param["n1"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(inpt)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 
x.shape # (96, 48, 48)
# 2
x = Conv2D(filters=param["n2"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 
x.shape # (128, 24, 24)
# 3
x = Conv2D(filters=param["n3"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 
x.shape # (192, 12, 12)
# 4
x = Conv2D(filters=param["n4"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x.shape # (256, 12, 12)
# 5
x = Conv2D(filters=param["n5"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = AveragePooling2D(pool_size=(param["apk"],param["apk"]), strides=param["aps"], padding="same")(x) # (256, 6, 6)
x.shape # (256, 6, 6)
# 6
x = Conv2D(filters=param["n6"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 7
x = Conv2D(filters=param["n7"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = AveragePooling2D(pool_size=(param["apk"],param["apk"]), strides=param["aps"], padding="same")(x) 
x.shape # (384, 3, 3)
# 8
x = Flatten()(x)
x = Dense(units=param["n8"], activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = Dropout(rate=0.5)(x)
# 9
x = Dense(units=param["n9"], activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = Activation("softmax")(x)
x.shape

model = Model(inputs=inpt, outputs=x)
#model.summary()
#plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

#model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1, momentum=0.9, decay=0.1), metrics=["accuracy"])
bs = 128; epc = 60; dcy = 0.04
0.1*(1-dcy)**np.arange(epc)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), metrics=["accuracy"]) # decay=dcy
early_stopping = EarlyStopping(monitor="loss", patience=2, mode='auto', verbose=1) # 在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练
#reduce = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=2, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0001, verbose=1)
t0 = pd.Timestamp.now()
model_fit = model.fit(x_train, y_train_ot, batch_size=bs, epochs=epc, verbose=1, callbacks=[early_stopping])
# validation_data=(x_valid, y_valid_ot); validation_split=0.2
t1 = pd.Timestamp.now()
print("time total spend: " + str(t1-t0))
print(model_fit.history)
print(model_fit.history.keys())
print(model_fit.history['loss'])
plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['acc'])

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

loss_train, acc_train = model.evaluate(x_train, y_train_ot, batch_size=bs, verbose=1)
print(loss_train, acc_train)
loss_test, acc_test = model.evaluate(x_test, y_test_ot, verbose=1)
print(loss_test, acc_test)

output = model.predict(x_test, batch_size=bs, verbose=1)
y_pred = np.argmax(output, axis=1)
sum(y_pred == y_test) / len(y_test) # 0.937
pd.crosstab(y_test, y_pred, margins=True)

# 模型和权重
model.save("trainSet_140_9.h5", overwrite=True, include_optimizer=True)
model = load_model('trainSet_140.h5', compile=True)
# 模型
json_string = model.to_json()
model = model_from_json(json_string)
yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
# 权重
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5', by_name=True)
