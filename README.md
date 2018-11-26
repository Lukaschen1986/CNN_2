# CNN_2

#### 卷积神经网络「失陷」，CoordConv来填坑  https://www.jiqizhixin.com/articles/uber-CoordConv  
#### 从FM推演各深度CTR预估模型(附代码)  https://www.jiqizhixin.com/articles/2018-07-16-17  

#### 使用GlobalAveragePooling2D是个明智的选择，相比Flatten，GlobalAveragePooling2D可以大量减少模型参数，降低过拟合的风险，同时显著降低计算成本，这也是现在主流的一些CNN架构的做法。  https://blog.csdn.net/oppo62258801/article/details/77930246/  
#### 将MaxPooling2D提至BatchNormalization和Activation前和放在它们后面是等价的，但是放在前面可以减少模型运算量。  
#### 除此之外，我更推荐你尝试Xception，它在众多图像识别领域中拔得头筹。在本项目的预测任务中，它能够轻松达到85%以上的测试集合准确率。  
#### 当前流行的一些优化器算法的优劣比较  http://ruder.io/optimizing-gradient-descent/  
#### 数据增强  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html  
#### ImageNet  https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/  
#### http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/  
#### Display Deep Learning Model Training History in Keras  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/  
#### 推出一个半月，斯坦福SQuAD问答榜单前六名都在使用BERT https://www.jiqizhixin.com/articles/2018-11-26-4  
#### 微软亚洲研究院：NLP将迎来黄金十年 https://www.jiqizhixin.com/articles/2018-11-25  
#### 自然语言处理中的语言模型预训练方法 https://www.jiqizhixin.com/articles/2018-10-22-3?from=synced&keyword=NLP  
#### Dropout 可能要换了 https://www.jiqizhixin.com/articles/112501  
#### DeepMind推出深度学习与强化学习进阶课程 https://www.jiqizhixin.com/articles/2018-11-24-3  
#### 中文项目：快速识别验证码 https://www.jiqizhixin.com/articles/2018-11-24  
#### 读懂智能对话系统（1）任务导向型对话系统 https://www.jiqizhixin.com/articles/2018-11-23-17  

#### 以下是我对改进模型提出的建议，希望对你有帮助：  
###### 1、模型融合（Model Ensembling）
###### 通过利用一些机器学习中模型融合的技术，如voting、bagging、blending以及staking等，可以显著提高模型的准确率与鲁棒性，且几乎没有风险。你可以参考我整理的机器学习笔记中的Ensemble部分。
###### 2、更多的数据
###### 对于深度学习（机器学习）任务来说，更多的数据意味着更为丰富的输入空间，可以带来更好的训练效果。我们可以通过数据增强（Data Augmentation）、对抗生成网络（Generative Adversarial Networks）等方式来对数据集进行扩充，同时这种方式也能提升模型的鲁棒性。
###### 3、更换人脸检测算法
###### 尽管OpenCV工具包非常方便并且高效，Haar级联检测也是一个可以直接使用的强力算法，但是这些算法仍然不能获得很高的准确率，并且需要用户提供正面照片，这带来的一定的不便。所以如果想要获得更好的用户体验和准确率，我们可以尝试一些新的人脸识别算法，如基于深度学习的一些算法。
###### 4、多目标监测
###### 更进一步，我们可以通过一些先进的目标识别算法，如RCNN、Fast-RCNN、Faster-RCNN或Masked-RCNN等，来完成一张照片中同时出现多个目标的检测任务。
