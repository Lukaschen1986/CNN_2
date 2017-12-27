# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
os.chdir("D:/my_project/Python_Project/test/deeplearning/HWDB1")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import cnn_layers_tf as clt

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
from keras import backend as K
K.image_data_format()
K.set_image_data_format('channels_first')
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model, model_from_json, model_from_yaml

f = open("./dataSet_tmp.txt", "rb")
dataSet_tmp = pickle.load(f); f.close()
x_train = dataSet_tmp["target"]; y_train = dataSet_tmp["label"]

N, H, W = x_train.shape
x_train = x_train.reshape(N, 1, H, W)
val_max = np.max(x_train)
x_train = (x_train/val_max).astype("float32")

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
y_train_ot = to_categorical(y_train)
y_test_ot = to_categorical(y_test)

inpt = Input(shape=(1, 96, 96))
# 1
x = Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(inpt)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (96, 48, 48)
# 2
x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (128, 24, 24)
# 3
x = Conv2D(filters=160, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (160, 12, 12)
# 4
x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 5
x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(x) # (256, 6, 6)
# 6
x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
# 7
x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
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
x = Dense(units=3755, activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = Activation("softmax")(x)
x.shape

model = Model(inputs=inpt, outputs=x)  
model.summary()
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1, momentum=0.9, decay=0.1), metrics=["accuracy"])

model.fit(x_train, y_train_ot, batch_size=128, epochs=2, validation_data=(x_test,y_test_ot), verbose=1)

## pred
loss_train, accu_train = model.evaluate(x_train, y_train_ot, verbose=1)
loss_valid, accu_valid = model.evaluate(x_valid, y_valid_ot, verbose=1)
loss_test, accu_test = model.evaluate(x_test, y_test_ot, verbose=1)

output = model.predict(x_test, batch_size=batch_size, verbose=1)
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
