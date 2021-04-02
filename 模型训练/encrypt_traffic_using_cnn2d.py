#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import struct
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adadelta

#**************************************读取由pcap格式转换成的idx1和idx3格式文件*****************
# 训练集文件
train_images_idx3_ubyte_file='train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file='train-labels-idx1-ubyte'
# 测试集文件
test_images_idx3_ubyte_file = 'test-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'test-labels-idx1-ubyte'
#**************************************读取由pcap格式转换成的idx1和idx3格式文件*****************



#*************************************数据预处理**************************************************
#***********************************************************************************************
#读取idx3（训练集）
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images

#解析标签
def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

#*数据处理
def process_images_labels(images,labels):
    x=images
    y=labels
    number=labels.size
    #对输入进行处理
    #每张图片为28*28*1
    x=images.reshape(number,28,28,1).astype('float32')
    x/=255#归一化
    
    #将输出变为onehot
    y=np_utils.to_categorical(y,12)
    return x,y
train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()
x_train,y_train=process_images_labels(train_images,train_labels)
x_test,y_test=process_images_labels(test_images,test_labels)
#************************************************************************************************
#**************************************数据预处理************************************************

#**************************************设置模型**************************************************
#************************************************************************************************
#设置模型
model=Sequential()
#卷积层1
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
#池化层1
model.add(MaxPooling2D(pool_size=(2,2)))
#卷积层2
model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu'))
#池化层2
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#全连接层1
model.add(Flatten())
model.add(Dense(units=1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=12,activation='softmax'))
#设定模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#开始训练
train_history=model.fit(x_train,y_train,validation_split=0.2,batch_size=100,epochs=30)
#查看效果
result=model.evaluate(x_train,y_train,batch_size=10000)
print('\nTrain Acc',result[1])
result=model.evaluate(x_test,y_test,batch_size=10000)
print('\nTest Acc',result[1])
#************************************************************************************************
#**************************************设置模型**************************************************


#*******************************使用训练好的数据预测模型*****************************************
#************************************************************************************************
#保存预测数据
prediction=model.predict(x_test)
#预测数据标签对应的类别
label2class={0:'Chat',1:'Email',2:'File',3:'P2P',4:'Streaming',5:'VoIP',6:'VPN-Chat',7:'VPN-Email',
            8:'VPN-File',9:'VPN-P2P',10:'VPN-Streaming',11:'VPN-VoIP'}

def predict(dates,n):
    #预测的数据为0~11标签，转换成对应类别
    for i in range(n):
        k=dates[i].argmax()
        print(label2class[k])
        
print("预测：")
predict(prediction,20)
print("实际：")
predict(y_test,20)
#*******************************使用训练好的数据预测模型*****************************************
#************************************************************************************************

