# -*- coding: utf-8 -*-
# https://zhuanlan.zhihu.com/p/33910138
# https://github.com/walid0925/AI_Artistry
import os
os.chdir("D:/my_project/Python_Project/deep_learning/neural_style")
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
K.image_data_format()
#K.set_image_data_format('channels_last')
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
import time
#from keras.models import Model
#from keras.optimizers import SGD, Adam


cImPath = "D:/my_project/Python_Project/deep_learning/neural_style/DSC090352.jpg"
sImPath = "D:/my_project/Python_Project/deep_learning/neural_style/timg.jpg"
genImOutputPath = "D:/my_project/Python_Project/deep_learning/neural_style/output.jpg"

targetHeight = 320
targetWidth = 480
targetSize = (targetHeight, targetWidth)

cImageOrig = Image.open(cImPath)
cImageSizeOrig = cImageOrig.size

cImage = load_img(path=cImPath, target_size=targetSize)
cImArr = img_to_array(cImage)
cImArr = preprocess_input(np.expand_dims(cImArr, axis=0))
cImArr = K.variable(cImArr, dtype="float32")

sImage = load_img(path=sImPath, target_size=targetSize)
sImArr = img_to_array(sImage)
sImArr = preprocess_input(np.expand_dims(sImArr, axis=0))
sImArr = K.variable(sImArr, dtype="float32")

gIm0 = np.random.randint(256, size=(targetWidth, targetHeight, 3)).astype("float64")
gIm0 = preprocess_input(np.expand_dims(gIm0, axis=0))
gImPlaceholder = K.placeholder(shape=(1, targetWidth, targetHeight, 3))

tf_session = K.get_session()
cModel = VGG16(include_top=False, weights="imagenet", input_tensor=cImArr)
#plot_model(cModel, to_file="cModel.png", show_shapes=True, show_layer_names=True)
cLayerName = ["block4_conv2"]
gModel = VGG16(include_top=False, weights="imagenet", input_tensor=gImPlaceholder)

def get_feature_reps(x, layer_names, model):
    featMatrices = []
    for layer in layer_names:
        featRaw = model.get_layer(layer).output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
        N_l = featRawShape[-1] # 卷积核数量
        M_l = featRawShape[1]*featRawShape[2] # H * W
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices
P = get_feature_reps(x=cImArr, layer_names=cLayerName, model=cModel)[0]


def get_content_loss(F, P):
    cLoss = 0.5 * K.sum(K.square(F - P))
    return cLoss


sLayerNames = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                ]

sModel = VGG16(include_top=False, weights="imagenet", input_tensor=sImArr)
As = get_feature_reps(x=sImArr, layer_names=sLayerNames, model=sModel)
ws = np.ones(len(sLayerNames)) / float(len(sLayerNames))

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.0)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss += w * 1/4 / (N_l**2 * M_l**2) * K.sum(K.square(G_gram - A_gram)) 
    return sLoss


def get_total_loss(gImPlaceholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=cLayerName, model=gModel)[0]
    Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss


def calculate_loss(gImArr):
    if gImArr.shape != (1, targetWidth, targetWidth, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
        
    loss_fcn = K.function(inputs=[gModel.input], 
                          outputs=[get_total_loss(gModel.input)])
    
    loss_get = loss_fcn([gImArr])[0].astype("float64")
    return loss_get


def calculate_grad(gImArr):
    if gImArr.shape != (1, targetWidth, targetHeight, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
        
    grad_fcn = K.function(inputs=[gModel.input], 
                          outputs=K.gradients(loss=get_total_loss(gModel.input), 
                                              variables=[gModel.input]))
    
    grad_get = grad_fcn([gImArr])[0].flatten().astype("float64")
    return grad_get


epc = 100
x_val = gIm0.flatten()
t0 = pd.Timestamp.now()
xopt, f_val, info = fmin_l_bfgs_b(func=calculate_loss, 
                                  x0=x_val, 
                                  fprime=calculate_grad, 
                                  maxiter=epc, 
                                  disp=True)
t1 = pd.Timestamp.now()
print(t1-t0)

#model = Model(inputs=[cImArr, sImArr], outputs=gIm0)
#
#bs = 128; epc = 100; lr = 0.1; dcy = 0.04
#lr*(1-dcy)**np.arange(epc)
#
#model.compile(loss="categorical_crossentropy", 
#              optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), 
#              metrics=["accuracy"]) # decay=dcy
#
#early_stopping = EarlyStopping(monitor="loss", patience=2, mode='auto', verbose=1)
#model_fit = model.fit(x_train, y_train_ot, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (targetWidth, targetHeight, 3):
        x = x.reshape((targetWidth, targetHeight, 3))
        
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    
    return x

#def reprocess_array(x):
#    x = np.expand_dims(x.astype('float64'), axis=0)
#    x = preprocess_input(x)
#    return x

def save_original_size(x, target_size=cImageSizeOrig):
    xIm = Image.fromarray(x)
    xIm = xIm.resize(target_size)
    xIm.save(genImOutputPath)
    return xIm

xOut = postprocess_array(xopt)
xIm = save_original_size(xOut)
print('Image saved')
