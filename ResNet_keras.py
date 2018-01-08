# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
#os.chdir("G:/DataSet/detail")
os.chdir("D:/my_project/Python_Project/test/deeplearning/HWDB1")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, concatenate
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

inpt = Input(shape=(param["c0"], param["h0"], param["w0"])) # a1
# 1
x = Conv2D(filters=param["n1"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(inpt)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x) # z2
x = PReLU(alpha_initializer=initializers.zeros())(x) # a2
# 2
x = Conv2D(filters=param["n2"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x) # z3
x = concatenate([x, inpt], axis=1) # z3 + a1
x = PReLU(alpha_initializer=initializers.zeros())(x) # a3
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 
# 3
x_resid = x # a1
