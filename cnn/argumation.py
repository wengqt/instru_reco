import cv2
import random








def img_rotate(src,ind):
    img = cv2.imread(src)
    rows, cols, channel = img.shape
    a= random.random()
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), int(30*a), 1.0)
    print(a)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite('./new/im2'+str(ind)+'.jpg',dst)




def up_down(src,ind):
    img = cv2.imread(src)
    v_flip = cv2.flip(img, 0)
    cv2.imwrite('./new/im0' + str(ind) + '.jpg', v_flip)


def resize(src,ind):
    img = cv2.imread(src)
    rows, cols, channel = img.shape
    if cols>100 and cols<200:
        img = cv2.resize(img,(int(cols/2),int(rows/2)))
    elif cols>200 and cols<300:
        img = cv2.resize(img, (int(cols / 3), int(rows / 3)))
    elif cols>300 and cols<400:
        img = cv2.resize(img, (int(cols / 4), int(rows / 4)))
    elif cols>400 :
        img = cv2.resize(img, (int(cols / 5), int(rows / 5)))
    cv2.imwrite('./new/neg' + str(ind) + '.jpg', img)



def resize_train(src,ind):
    img = cv2.imread(src)
    img = cv2.resize(img,(64,32))
    cv2.imwrite('../JPEGImages/image' + str(ind) + '.jpg', img)



def resize_ori(src,ind):
    img = cv2.imread(src)
    sh=img.shape
    # h = int(sh[0]*256/sh[1])
    img = cv2.resize(img, (256, 341))
    cv2.imwrite('./new/image' + str(ind) + '.jpg', img)



from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from glob import glob
import cv2
import numpy as np

# 图片生成器
def generator_img(src):
    datagen = ImageDataGenerator(
                rotation_range=5,
                width_shift_range=0.05,
                height_shift_range=0.05,
                rescale=1./255,
                # shear_range=0.05,
                zoom_range=[1.0,1.1],
                # horizontal_flip=True,
                fill_mode='constant')

    # 打印转换前的图片
    img = load_img(src)


    # 将图片转换为数组，并重新设定形状
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # x的形状重组为(1,width,height,channels)，第一个参数为batch_size

    # 这里人工设置停止生成， 并保存图片用于可视化
    i = 0
    for batch in datagen.flow(x,batch_size=1,save_to_dir='./9a',save_prefix='b',save_format='jpg'):
        i +=1
        if i > 10:
            return

for i in range(0,320):
    path='./cnn_num/0/a'+str(i)+'.jpg'
    generator_img(path)


# for i in range(0,8):
#     # path ='../ssd_trains/JPEGImages/image'+str(i)+'.jpg'
#     # path ='./new/image'+str(i)+'.png'
#     # # generator_img(path)
#     # img = cv2.imread(path)
#     # cv2.imwrite('./new/1/image'+str(i)+'.jpg',img)
#     path = '../train_/-1/a_'+str(i)+'.jpg'
#     generator_img(path)


# for j in range(-1,10):
#     for i in range(0,1140):
#         path1 = '../train_/' + str(j) + '/_' + str(i) + '.jpg'
#         img = cv2.imread(path1)
#         img = cv2.resize(img,(32,64))
#         path2 = '../cnn_train/' + str(j) + '/' + str(i) + '.jpg'
#         cv2.imwrite(path2,img)