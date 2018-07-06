# -*- coding: utf-8 -*-
# https://zhuanlan.zhihu.com/p/33910138
# https://github.com/walid0925/AI_Artistry
import os
os.chdir("D:/my_project/Python_Project/iTravel/room_mapping/txt")
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
K.image_data_format()
#K.set_image_data_format('channels_last')
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from scipy.optimize import fmin_l_bfgs_b
import time

cImPath = "D:/my_project/Python_Project/iTravel/room_mapping/txt/cat.jpg"
sImPath = "D:/my_project/Python_Project/iTravel/room_mapping/txt/violin_and_palette.jpg"
genImOutputPath = ""

targetHeight = 512
targetWidth = 512
targetSize = (targetHeight, targetWidth)

cImageOrig = Image.open(cImPath)
cImageSizeOrig = cImageOrig.size

cImage = load_img(path=cImPath, target_size=targetSize)
cImArr = img_to_array(cImage)
cImArr = K.variable(preprocess_input(np.expand_dims(cImArr, axis=0)), dtype="float32")

sImage = load_img(path=sImPath, target_size=targetSize)
sImArr = img_to_array(sImage)
sImArr = K.variable(preprocess_input(np.expand_dims(sImArr, axis=0)), dtype="float32")

gIm0 = np.random.randint(256, size=(targetWidth, targetHeight, 3)).astype("float64")
gIm0 = preprocess_input(np.expand_dims(gIm0, axis=0))
gImPlaceholder = K.placeholder(shape=(1, targetWidth, targetHeight, 3))

tf_session = K.get_session()
cModel = VGG16(include_top=False, weights="imagenet", input_tensor=cImArr)
cLayerName = "block4_conv2"
gModel = VGG16(include_top=False, weights="imagenet", input_tensor=gImPlaceholder)

def get_feature_reps(x, layer_names, model):
    featMatrices = []
    for layer in layer_names:
        featRaw = model.get_layer(layer).output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices
P = get_feature_reps(x=cImArr, layer_names=[cLayerName], model=cModel)[0]
F = get_feature_reps(x=gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]

def get_content_loss(F, P):
    cLoss = 0.5 * K.sum(K.square(F - P))
    return cLoss
contentLoss = get_content_loss(F, P)

sLayerNames = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                #'block5_conv1'
                ]

sModel = VGG16(include_top=False, weights="imagenet", input_tensor=sImArr)
Gs = get_feature_reps(x=gImPlaceholder, layer_names=sLayerNames, model=gModel)
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
styleLoss = get_style_loss(ws, Gs, As)
alpha = 1.0; beta = 10000.0
totalLoss = alpha * contentLoss + beta * styleLoss

calculate_loss = K.function(inputs=[gModel.input], outputs=totalLoss)
get_loss = calculate_loss[0].astype("float64")

calculate_grad = K.function(inputs=[gModel.input], outputs=K.gradients(loss=totalLoss, variables=[gModel.input]))
get_grad = calculate_grad[0].flatten().astype("float64")


iterations = 600
x_val = gIm0.flatten()
start = time.time()
xopt, f_val, info = fmin_l_bfgs_b(func=get_loss, x0=x_val, fprime=get_grad, maxiter=iterations, disp=True)

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
end = time.time()
print('Time taken: {}'.format(end-start))
