import numpy as np
import os
import cv2
from PIL import Image
# import splitNumber


class Softmax:
    def loadData(self, dir):    #给出文件目录，读取数据
        digits = list() #数据集（数据）
        labels = list() #标签
        if os.path.exists(dir): #判断目录是否存在
            for i in range(-1,10):
                files = os.listdir(dir+'/'+str(i)) #获取目录下的所有文件名
                for file in files:  #遍历所有文件
                    if file.split('_')[0]!='.DS':
                        labels.append(file.split('_')[0])   #按照文件名规则，文件名第一位是标签
                        # print(file.split('_')[0])
                    with open(dir+'/'+str(i) +'/'+ file) as f:  #通过“目录+文件名”，获取文件内容

                        try:
                            img = Image.open(dir + '/' + str(i) + '/' + file)
                            img = img.convert('L')
                            width, hight = img.size
                            img = np.asarray(img, dtype='float64') / 256.

                            tmp = img.reshape(1, hight * width)[0]
                            # print(tmp)
                            # digit = list()
                            # for line in f:  #遍历文件每一行
                            #     digit.extend(map(int, list(line.replace('\n', ''))))    #遍历每行时，把数字通过extend的方法扩展
                            digits.append(tmp)  # 将数据扩展进去
                        except IOError:
                            print("Error: 读取文件失败",dir+'/'+str(i) +'/'+ file)



        digits = np.array(digits)   #数据集

        labels = list(map(int, labels)) #标签
        labels = np.array(labels).reshape((-1, 1))  #将标签重构成(N, 1)的大小
        # print(labels)
        return digits, labels

    # def processImg(self, dir1):
    #     digits = list()
    #     dots=[]
    #     if os.path.exists(dir1):
    #         total,dots = v5_tempFinall.processImg(dir1)
    #         for one_list in total:
    #             digit = []
    #             for one_dir in one_list:
    #                 # print(one_dir)
    #                 try:
    #                     img = Image.open(one_dir)
    #                     img = img.convert('L')
    #                     width, hight = img.size
    #                     img = np.asarray(img, dtype='float64') / 256.
    #
    #                     tmp = img.reshape(1, hight * width)[0]
    #                     # print(tmp)
    #                     # digit = list()
    #                     # for line in f:  #遍历文件每一行
    #                     #     digit.extend(map(int, list(line.replace('\n', ''))))    #遍历每行时，把数字通过extend的方法扩展
    #                     digit.append(tmp)  # 将数据扩展进去
    #                 except IOError:
    #                     print("Error: 读取文件失败", one_dir)
    #             digits.append(digit)
    #
    #     return digits,dots



    def softmax(self, X):   #softmax函数
        X = X - np.max(X)
        exp_x = np.exp(X)
        return exp_x / np.sum(exp_x)

    def train(self, digits, labels, maxIter = 100, alpha = 0.1):
        self.weights = np.random.uniform(0, 1, (11, 8192))
        for iter in range(maxIter):
            for i in range(len(digits)):
                x = digits[i].reshape(-1, 1)
                y = np.zeros((11, 1))
                y[labels[i]] = 1
                y_ = self.softmax(np.dot(self.weights, x))
                self.weights -= alpha * (np.dot((y_ - y), x.T))
        return self.weights

    def predict(self, digit):   #预测函数
        # print(digit,np.dot(self.weights, digit),np.argmax(np.dot(self.weights, digit)))
        match = [0,1,2,3,4,5,6,7,8,9,-1]
        return match[np.argmax(np.dot(self.weights, digit))] #返回softmax中概率最大的值

if __name__ == '__main__':
    softmax = Softmax()
    trainDigits, trainLabels = softmax.loadData('./train')
    softmax.train(trainDigits, trainLabels, maxIter=100) #训练

