# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.chdir("D:/my_project/Python_Project/deep_learning/rnn")
from scipy.stats import itemfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.models.word2vec import Word2Vec # https://radimrehurek.com/gensim/models/word2vec.html

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
from keras import backend as K
K.image_data_format()
K.set_image_data_format("channels_first")
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model, model_from_json, model_from_yaml
#from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
#import gc
#from PIL import Image, ImageEnhance, ImageOps, ImageFile
#import cv2

# load data
data = pd.read_csv("./input/Combined_News_DJIA.csv")
data_train = data[data["Date"] < "2015-01-01"]
data_test = data[data["Date"] > "2014-12-31"]

# set corpus, train, test
train = data_train[data_train.columns[2:]]
corpus = train.values.flatten().astype(str) # 语料库
X_train = train.values.astype(str) # DataFrame to ndarray
X_train = np.array([" ".join(x) for x in X_train]) # 将每一列文本按行拼成一个长字符串

test = data_test[data_test.columns[2:]]
X_test = test.values.astype(str)
X_test = np.array([" ".join(x) for x in X_test])

y_train = data_train["Label"].values.astype(np.int32)
y_test = data_test["Label"].values.astype(np.int32)

# 分词
corpus = [word_tokenize(x, language="english") for x in corpus] 
X_train = [word_tokenize(x, language="english") for x in X_train] 
X_test = [word_tokenize(x, language="english") for x in X_test] 

# preprocessing
stop = stopwords.words("english") # 停止词
hasNumbers = lambda inputString: bool(re.search(r'\d', inputString)) # 数字
isSymbol = lambda inputString: bool(re.match(r'[^\w]', inputString)) # ^: 匹配不是[A-Za-z0-9_]的特殊符号
wordnet_lemmatizer = WordNetLemmatizer() # lemma 单词变体还原

def check(word):
    word = word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True
    
def preprocessing(sentence):
    res = []
    for word in sentence:
        if check(word):
            new_word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '') # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            new_word = wordnet_lemmatizer.lemmatize(new_word)
            res.append(new_word)
    return res   
    
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]    

# Word2Vec
w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
vocab = w2v_model.wv.vocab

def transform_to_matrix(x, w2v_model, maxlen=256, size=128): 
    res = []
    for sentence in x:
        matrix = []
        for i in range(maxlen):
            try:
                w2v = w2v_model[sentence[i]].tolist()
                matrix.append(w2v)
            except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                w2v = [0] * size
                matrix.append(w2v)
        res.append(matrix)
    return res

X_train = transform_to_matrix(X_train, w2v_model, maxlen=256, size=128)
X_test = transform_to_matrix(X_test, w2v_model, maxlen=256, size=128)

X_train = np.array(X_train)
X_test = np.array(X_test)

_, H, W = X_train.shape
X_train = X_train.reshape(X_train.shape[0], 1, H, W)
X_test = X_test.reshape(X_test.shape[0], 1, H, W)

# CNN
param = {"ck":3, "cs":1, 
         "mpk":2, "mps":2, 
         "apk":2, "aps":2, 
         "speed":4, "reg":0.01,
         "h0":256, "w0":128, "c0":1, 
         "n1":16, "n2":32, "n3":64,
         "n4":128, "n5":256, "n6":512,
         "n7":512, "n8":1024, "n9":512,
         "n10":1}


inpt = Input(shape=(param["c0"], param["h0"], param["w0"]))
x = Conv2D(filters=param["n1"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(inpt)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 

x = Conv2D(filters=param["n2"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 

x = Conv2D(filters=param["n3"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x) 

x = Conv2D(filters=param["n4"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)

x = Conv2D(filters=param["n5"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = MaxPooling2D(pool_size=(param["mpk"],param["mpk"]), strides=param["mps"], padding="same")(x)

x = Conv2D(filters=param["n6"], kernel_size=(param["ck"],param["ck"]), strides=param["cs"], padding="same", activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), kernel_regularizer=l2(param["reg"]))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)

x = Flatten()(x)
x = Dense(units=param["n8"], activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = Dropout(rate=0.5)(x)

x = Dense(units=param["n9"], activation=None, use_bias=False, kernel_initializer=initializers.TruncatedNormal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
x = PReLU(alpha_initializer=initializers.zeros())(x)
x = Dropout(rate=0.5)(x)

x = Dense(units=param["n10"], activation=None, use_bias=False, kernel_initializer=initializers.random_normal(0.0, 0.01))(x)
x = BatchNormalization(axis=1, center=True, beta_initializer=initializers.zeros(), scale=True, gamma_initializer=initializers.ones(), epsilon=10**-8, momentum=0.9)(x)
#x = Activation("softmax")(x)
x = Activation("sigmoid")(x)

cnn_model = Model(inputs=inpt, outputs=x)
cnn_model.summary()
plot_model(cnn_model, to_file="cnn_model.png", show_shapes=True, show_layer_names=True)

bs = 128; epc = 10; lr = 0.1; dcy = 0.04
lr*(1-dcy)**np.arange(epc)

cnn_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), metrics=["accuracy"]) # ,decay=dcy, loss="binary_crossentropy", "categorical_crossentropy"

early_stopping = EarlyStopping(monitor="loss", patience=2, mode="auto", verbose=1) # 在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练

t0 = pd.Timestamp.now()
model_fit = cnn_model.fit(X_train, y_train, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])
# validation_data=(x_valid, y_valid_ot); validation_split=0.2
t1 = pd.Timestamp.now()

print("time total spend: " + str(t1-t0))
print(model_fit.history)
print(model_fit.history.keys())
print(model_fit.history['loss'])
plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['acc'])

#X_test, y_test = shuffle(X_test, y_test, random_state=0)
loss_test, acc_test = cnn_model.evaluate(X_test, y_test, verbose=1)
print(loss_test, acc_test)

output = cnn_model.predict(X_test, batch_size=bs, verbose=1)
y_pred = np.argmax(output, axis=1)
sum(y_pred == y_test) / len(y_test) # 0.937
pd.crosstab(y_test, y_pred, margins=True)

# 模型和权重
cnn_model.save("cnn_model.h5", overwrite=True, include_optimizer=True)
cnn_model = load_model('cnn_model.h5', compile=True)
# 模型
json_string = cnn_model.to_json()
cnn_model = model_from_json(json_string)
yaml_string = cnn_model.to_yaml()
cnn_model = model_from_yaml(yaml_string)
# 权重
cnn_model.save_weights('my_model_weights.h5')
cnn_model.load_weights('my_model_weights.h5', by_name=True)
