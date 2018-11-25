from __future__ import absolute_import,division,print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import ssl
from PIL import Image
import numpy as np
import cv2

ssl._create_default_https_context = ssl._create_unverified_context


batch_size = 128
num_classes = 11
epochs = 12

# input image dimensions
img_rows, img_cols = 64, 32





input_shape = (img_rows, img_cols, 1)

def load_image():
    train_img =[]
    train_label=[]
    test_img=[]
    test_label=[]
    with open('../cnn_train/train.txt') as f:
        for line in f.readlines():
            pos = line.split(' ')[0]
            index = int(line.split(' ')[1])
            img = cv2.imread('../cnn_train/' + pos,0)
            # print(img)
            train_img.append(img)
            train_label.append(index)
            # img.close()
    f.close()
    with open('../cnn_train/val.txt') as f:
        for line in f.readlines():
            pos = line.split(' ')[0]
            index = int(line.split(' ')[1])
            img = cv2.imread('../cnn_train/' + pos,0)
            test_img.append(img)
            test_label.append(index)
            # img.close()
    f.close()
    return np.array(train_img) ,np.array(train_label) ,np.array(test_img) ,np.array(test_label)

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = load_image()
#
# print(x_train[0])
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# #TFRcord文件
# TRAIN_FILE = './cnn/train32.tfrecords'
# VALIDATION_FILE = './cnn/val32.tfrecords'
#
# #图片信息
# NUM_CLASSES = 10
# IMG_HEIGHT = 64
# IMG_WIDTH = 32
# IMG_CHANNELS = 1
# IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS
#
# # NUM_TRAIN = convert_to_tfrecords.NUM_TRAIN
# # NUM_VALIDARION = convert_to_tfrecords.NUM_VALIDARION
#
# def read_and_decode(filename_queue):
#     #创建一个reader来读取TFRecord文件中的样例
#     reader = tf.TFRecordReader()
#     #从文件中读出一个样例
#     _,serialized_example = reader.read(filename_queue)
#     #解析读入的一个样例
#     features = tf.parse_single_example(serialized_example,features={
#         'label':tf.FixedLenFeature([],tf.int64),
#         'image_raw':tf.FixedLenFeature([],tf.string)
#         })
#     #将字符串解析成图像对应的像素数组
#     image = tf.decode_raw(features['image_raw'],tf.uint8)
#     label = tf.cast(features['label'],tf.int32)
#
#     image.set_shape([IMG_PIXELS])
#     image = tf.reshape(image,[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
#     image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
#
#     return image,label
#
# #用于获取一个batch_size的图像和label
# def inputs(data_set,batch_size,num_epochs):
#     if not num_epochs:
#         num_epochs = None
#     if data_set == 'train':
#         file = TRAIN_FILE
#     else:
#         file = VALIDATION_FILE
#
#     with tf.name_scope('input') as scope:
#         filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
#     image,label = read_and_decode(filename_queue)
#     #随机获得batch_size大小的图像和label
#     images,labels = tf.train.shuffle_batch([image, label],
#         batch_size=batch_size,
#         num_threads=4,
#         capacity=1000 + 3 * batch_size,
#         min_after_dequeue=1000
#     )
#
#     return images,labels



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# x_train,y_train = inputs('train',6600,False)
# x_test,y_test = inputs('test',4000,False)

print(x_train,y_train)
print('x_train shape:', x_train.shape)
print(y_train.shape, 'label samples')
print(y_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(filepath='222_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
callbacks = [model_checkpoint]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          # steps_per_epoch=50,
          # validation_steps=40,
          callbacks=callbacks,
          validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
model.save('myCNN_pointer2.h5')