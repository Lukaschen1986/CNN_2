# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
os.chdir("D:/my_project/Python_Project/test/deeplearning")
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageDraw
from keras.preprocessing import image
import face_recognition as fr
import cv2

hat = Image.open("hat.png", mode="r") # 载入帽子图片
hat_array = image.img_to_array(hat, "channels_last").astype("uint8")
hat_array = cv2.flip(hat_array, flipCode=1)
r,g,b,a = cv2.split(hat_array)

hat_rgb = cv2.merge((r,g,b))
Image.fromarray(hat_rgb)
cv2.imwrite("hat_alpha.jpg", a)

#body = Image.open("cwd.jpg", mode="r")
#body = body.resize((600, 800), Image.BICUBIC)
#body.save("cwd.jpg")

body = fr.load_image_file("cwd.jpg") # 载入图片
face_loc = fr.face_locations(body, model="hog") # 识别人脸
row_up, col_right, row_down, col_left = face_loc[0] # 标记四角
cv2.rectangle(body, (col_left, row_up), (col_right, row_down), (0, 0, 255), 2) # 框出人脸
Image.fromarray(body)

face_landmarks_list = fr.face_landmarks(body, face_locations=face_loc)[0] # 识别五官
left_eyebrow = face_landmarks_list["left_eyebrow"][0]
right_eyebrow = face_landmarks_list["right_eyebrow"][0]
center_eyebrow = ((left_eyebrow[0]+right_eyebrow[0])//2, (left_eyebrow[1]+right_eyebrow[1])//2) # 计算两眼中心点坐标

# 根据人脸大小调整帽子大小
factor = 1.5
w = col_right - col_left
resized_hat_h = int(round(hat_rgb.shape[0] * w / hat_rgb.shape[1] * factor))
resized_hat_w = int(round(hat_rgb.shape[1] * w / hat_rgb.shape[1] * factor))

if resized_hat_h > row_up:   
    resized_hat_h = row_up-1

hat_resize = cv2.resize(hat_rgb, (resized_hat_w, resized_hat_h))

# 用alpha通道作为mask   
mask = cv2.resize(a, (resized_hat_w, resized_hat_h))   
mask_inv = cv2.bitwise_not(mask) # 颜色取反
'''
去Alpha通道作为mask。并求反。这两个mask一个用于把帽子图中的帽子区域取出来，一个用于把人物图中需要填帽子的区域空出来。
'''

dh = 60; dw = 150
body_2 = fr.load_image_file("cwd.jpg") # 载入图片
bg_roi = body_2[row_up+dh-resized_hat_h:row_up+dh, row_up+dw:row_up+dw+resized_hat_w]
bg_roi = bg_roi.astype("float32")
mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
alpha = mask_inv.astype("float32")/255
alpha = cv2.resize(alpha, (bg_roi.shape[1], bg_roi.shape[0]))
bg = cv2.multiply(alpha, bg_roi)
bg = bg.astype('uint8')

# 提取帽子区域
hat = cv2.bitwise_and(hat_resize, hat_resize, mask=mask)
# 添加圣诞帽, 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）   
hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))   
# 两个ROI区域相加   
add_hat = cv2.add(bg,hat)   
# cv2.imshow("add_hat",add_hat) 
# 把添加好帽子的区域放回原图
#body[(row_up+dh-resized_hat_h):(row_up+dh),(center_eyebrow[0]-resized_hat_w//3):(center_eyebrow[0]+resized_hat_w//3*2)] = add_hat

body_2 = fr.load_image_file("cwd.jpg") # 载入图片
body_2[row_up+dh-resized_hat_h:row_up+dh, row_up+dw:row_up+dw+resized_hat_w] = add_hat
body_with_hat = Image.fromarray(body_2)
body_with_hat.save("cwd_with_hat.jpg")
