# 读取句子图片，按字切分，矩阵变换
pic = Image.open("sent.png", mode="r")
pic_array = image.img_to_array(pic, "channels_last").astype("uint8") # img_to_array
pic_grey = cv2.cvtColor(pic_array, code=cv2.COLOR_BGR2GRAY) # 灰度化
pic_grey = cv2.copyMakeBorder(pic_grey, 2, 2, 2, 2, cv2.BORDER_REPLICATE) # Add padding around the image
pic_threshold = cv2.threshold(pic_grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1] # 二值化 与 黑白翻转
Image.fromarray(pic_threshold)
pic_contours = cv2.findContours(pic_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
'''
第一个参数是寻找轮廓的图像；
第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
    cv2.RETR_EXTERNAL表示只检测外轮廓
    cv2.RETR_LIST检测的轮廓不建立等级关系
    cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    cv2.RETR_TREE建立一个等级树结构的轮廓。
第三个参数method为轮廓的近似办法
    cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
    cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
'''
contours = pic_contours[1]

image_regions = []
for contour in contours:
    # contour = contours[1]
    (x, y, w, h) = cv2.boundingRect(contour)
    if w / h > 1.25:
        half_width = int(w / 2)
        image_regions.append((x, y, half_width, h))
        image_regions.append((x + half_width, y, half_width, h))
    else:
        image_regions.append((x, y, w, h))

image_regions = sorted(image_regions, key=lambda x: x[0])

for bounding_box in image_regions:
    # (x, y, w, h) = image_regions[13]
    (x, y, w, h) = bounding_box
    single_pic = pic_threshold[y - 2:y + h + 2, x - 2:x + w + 2]
    Image.fromarray(single_pic)

# 手工处理
img_size = 96
x_predict = np.zeros((img_size, img_size), dtype="uint8")
x_predict = np.expand_dims(x_predict, axis=0)

file_names = ["yong.png","wo.png","yi.png","sheng.png","ni.png","nian.png","tian.png","wu.png"]
#file_names = ["hua.png","wei.png","hui.png","tong.png","shang.png","lv.png"]
for file_name in file_names:
    pic = Image.open(file_name, mode="r")  
    pic_resize = pic.resize((img_size, img_size), Image.BICUBIC)
    pic_array = image.img_to_array(pic_resize, "channels_last").astype("uint8")
    pic_grey = cv2.cvtColor(pic_array, code=cv2.COLOR_BGR2GRAY)
    pic_threshold = cv2.threshold(pic_grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    pic_new = np.expand_dims(pic_threshold, axis=0)
    x_predict = np.concatenate((x_predict, pic_new), axis=0)

x_predict = x_predict[1:]
#sample = x_predict[0]
#Image.fromarray(sample)

# predict
#x_predict = pic_new
N, H, W = x_predict.shape
x_predict = x_predict.reshape(N, 1, H, W)
val_max = 255
x_predict = (x_predict/val_max).astype("float32")

output = model.predict(x_predict, verbose=1)
output_df = pd.DataFrame(output, columns=dicts["describe"])
y_pred = np.argmax(output, axis=1)

dicts = pd.DataFrame({"label":list(set(y_test)), "describe":z_test})
dicts.iloc[y_pred.tolist(),0].tolist()
