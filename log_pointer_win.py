

import sys


import cv2
import numpy as np
import math
import peakdetective
from keras.models import load_model
import platform

from keras import backend as K

import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernel6 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))





IMG_ERR=1
CIRCLE_ERR=2
MODEL_ERR=3
getScaleArea_ERR=5
convertPolar_ERR=6
NUM_ERR=7

def imwrite(_path,_img):
    cv2.imencode('.jpg',_img)[1].tofile(_path)



def findContours(findContoursImg,dstCanny,heartsarr=None,dst=None,offset1=0,offset2=0,big_index=0):
    """
    findContours
    findContoursImg 需要边缘检测的图片
    dstCanny 需要切割的图1
    heartsarr 圆心点位置
    dst 需要切割的图2
    offset1 矩形右上点横纵坐标的偏移
    offset2 矩形长宽的延长
    """
    _, contours, hierarchy = cv2.findContours(findContoursImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours)==0:
        raise Exception('最大区域矩形检测失败！', 'line 42 in function findContours')



    c = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(c[big_index])
    new_img =None
    if dst is not None:
        # cv2.rectangle(dst, (x-offset1, y-offset1), (x + w+offset2, y + h+offset2), (0, 255, 0), 2)
        new_img = dst[y-offset1:y + h+offset2,x-offset1:x + w+offset2]
    new_img_canny = dstCanny[y-offset1:y + h+offset2,x-offset1:x + w+offset2]
    imwrite(dir_path+'/cut_findContours.jpg', new_img_canny)
    _heart= None
    if heartsarr is not None:
        _heart = heartsarr.copy()
        # print(heartsarr)
        for a in _heart:
            a[0] =a[0]-x+offset1
            a[1] = a[1] - y + offset1
    return new_img,new_img_canny,_heart





def findHearts(src,org):
    '''

    :param src: 灰度图，用于houghcircle
    :param org: 原图，彩色图
    :return: hearts 按半径排序的圆
    '''

    """
    找刻度轴的圆心
    image为输入图像，需要灰度图
    method为检测方法,常用CV_HOUGH_GRADIENT
    dp为检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，如dp=1，累加器和输入图像具有相同的分辨率，如果dp=2，累计器便有输入图像一半那么大的宽度和高度
    minDist表示两个圆之间圆心的最小距离
    param1有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半
    param2有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值，它越小，就越可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了
    minRadius有默认值0，圆半径的最小值
    maxRadius有默认值0，圆半径的最大值
    """

    # src = cv2.erode(src,kernel3)
    # src = cv2.equalizeHist(src)
    imwrite(dir_path+'/2_cut_find_heart.jpg',src)
    print('img',dir_path+'/2_cut_find_heart.jpg')
    circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=60, minRadius=50,maxRadius=0)
    try:

        # print(len(circles2[0]))
        if len(circles2[0])<3 or circles2 is None:
            raise Exception("识别到的圆 少于 3")

        circles2 = np.uint16(np.around(circles2))
    except Exception as err:
        print('err',err)
        circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=60, param2=60, minRadius=50, maxRadius=0)
        circles2 = np.uint16(np.around(circles2))

    if len(circles2[0])<3 or circles2 is None:
        print('err','未识别到准确的圆')
        sys.exit(CIRCLE_ERR)

    cuto_cp = org.copy()
    for i in circles2[0,:]:
        # draw the outer circle
        cv2.circle(cuto_cp,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cuto_cp,(i[0],i[1]),2,(0,0,255),3)


    for i in range(0,3):
        onecircle = circles2[0][i]
        cv2.circle(cuto_cp,(onecircle[0],onecircle[1]),onecircle[2],(0,0,255),2)
        # draw the center of the circle
        cv2.circle(cuto_cp,(onecircle[0],onecircle[1]),2,(0,0,255),3)

    imwrite(dir_path+'/2_cut_Circles.jpg',cuto_cp)
    print('img', dir_path+'/2_cut_Circles.jpg')
    '''
    找到圆心位置之后，可以进行极坐标转换：
    参考 https://zhuanlan.zhihu.com/p/30827442
    '''
    # polar = cv2.logPolar(eroded, (circles2[0][1][0], circles2[0][1][1]), 150, cv2.WARP_FILL_OUTLIERS)
    # polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # imwrite('./findpointer/cut_polar.jpg', polar)

    hearts = circles2[0][:3]
    if hearts[1][2] < hearts[2][2]:
        tmp = hearts[1].copy()
        hearts[1] = hearts[2]
        hearts[2] = tmp

    return hearts




def get_border(arr,delta,d2=10):
    '''

    :param arr: input array
    :param delta: the min range between arr item
    :return: border
    '''
    getTwo=False
    border=[]
    for i in range(len(arr)):
        if getTwo==False:
            if arr[i] >=d2:
                border.append(i)
                getTwo = True
        else:
            if arr[i] <d2:
                if i>border[len(border)-1]+delta:
                    border.append(i)
                    getTwo = False
                else:
                    del border[len(border)-1]
                    getTwo = False
            if i>=len(arr)-1:
                border.append(i)

    return border





def getLineBorder(src,min_range=0):
    (h1, w1) = src.shape
    horizon = [0 for z in range(0, h1)]

    for i in range(0, h1):  # 遍历一行
        for j in range(0, w1):  # 遍历一列
            if src[i, j] != 0:
                horizon[i] += 1

    # line_border = []

    border_arr = get_border(horizon, min_range,p3)


    return border_arr,horizon







def projectVertical(src):
    (h1, w1) = src.shape
    vertical = [0 for z in range(0, w1)]

    for i in range(0, w1):  # 遍历一lie
        for j in range(0, h1):  # 遍历一hang
            if src[j, i] != 0:
                vertical[i] += 1
    return vertical




def getEachNum(src,delta=0):
    (h1, w1) = src.shape
    vertical = projectVertical(src)
    line_border = []

    border_arr = get_border(vertical, delta,p4)


    return border_arr, vertical




def cutNums(nums_Arr,cut,path):
    cut_num = []
    for i in range(0,len(nums_Arr),2):
        new_cut = cut[:,nums_Arr[i]:nums_Arr[i+1]]
        cut_num.append(new_cut)
        # new_cut = cv2.erode(new_cut, kernel5)
        imwrite(path+str(int(i/2))+'.jpg', new_cut)
        # print('img', path+str(int(i/2))+'.jpg')

    return cut_num






# def learnNums():
#     softmax_learn = softmax.Softmax()
#     trainDigits, trainLabels = softmax_learn.loadData('./train')
#     softmax_learn.train(trainDigits, trainLabels, maxIter=100)  # 训练
#     return softmax_learn




def findMainZone(path):
    if platform.system() == 'Windows':
        # path = path.encode('utf-8')
        # path = path.decode()
        img1 = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)
        # img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    else:
        img1 = cv2.imread(path)
    try:
        if img1 is None:
            raise Exception('找不到图片! 图片路径有误。')
    except Exception as err:
        # print(err)
        print('err', err)
        sys.exit(1)
    img1shape = img1.shape
    img1 = cv2.resize(img1, (int(img1shape[1]/2), int(img1shape[0]/2)))


    # canny = cv2.Canny(cv2.GaussianBlur(img1, (7, 7), 0), 100, 250)
    # imwrite(dir_path+'/canny.jpg', canny)

    canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    # canny = cv2.cvtColor(cv2.bilateralFilter(img1, 9, 70, 70), cv2.COLOR_BGR2GRAY)
    '''
    找到仪表区域
    '''
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=p1, param2=p2, minRadius=50)
    if circles is None:
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=70, param2=50, minRadius=50)

    circles = np.uint16(np.around(circles))[0]
    img1_cp = img1.copy()
    for i in circles[0:]:
        # 画圆圈
        cv2.circle(img1_cp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img1_cp, (i[0], i[1]), 2, (0, 0, 255), 3)

    # _circles = circles[0,:3] #投票数最大的圆
    # _circles = sorted(_circles,key=lambda x:x[2],reverse=True )
    # print(_circles)
    the_circle = circles[0]
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), the_circle[2], (0, 0, 255), 2)
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), 2, (0, 0, 255), 3)
    imwrite(dir_path+'/1_circles.jpg', img1_cp)
    print('img', dir_path+'/1_circles.jpg')
    # print(the_circle[1],the_circle[2])
    # print(int(the_circle[1]) - int(the_circle[2]))
    d1 =0 if int(the_circle[1]) - int(the_circle[2])<0 else int(the_circle[1]) - int(the_circle[2])
    d2 =0 if int(the_circle[0]) - int(the_circle[2])<0 else int(the_circle[0]) - int(the_circle[2])
    cutImg_o = img1[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    gray_o = canny[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    imwrite(dir_path+'/1_cut.jpg', cutImg_o)
    print('img', dir_path+'/1_cut.jpg')

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    # cut_g = cv2.GaussianBlur(cutImg, (7, 7), 1)
    # cut_g = cv2.equalizeHist(cut_g)
    # cutCanny = cv2.Canny(cut_g, 50, 100)
    # imwrite(dir_path+'/1_cut_Canny.jpg', cutCanny)
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg1 = cv2.bilateralFilter(cutImg, 9, 70, 70)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),0) #灰度图，用于后面的圆圈识别
    kernel = np.ones((3, 3), np.float32) / 25
    cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别

    cutImg3 = cv2.adaptiveThreshold(cutImg2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    # ret2, cutImg3 = cv2.threshold(cutImg2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imwrite(dir_path+'/1_cut_adapt.jpg', cutImg3)
    print('img', dir_path+'/1_cut_adapt.jpg')
    return cutImg3,cutImg_o,cutImg1,gray_o



def findPointer2(_img,_heart):
    '''

    :param _img: 二值图
    :param _heart: 两个刻度盘的圆心
    :return: 无指针图
    '''
    # _img = cv2.resize(_img,(500,500))
    _shape = _img.shape
    _img1 = _img.copy()
    org = _img.copy()
    # _img1 = cv2.erode(_img1, kernel3, iterations=1)
    # _img1 = cv2.dilate(_img1, kernel3, iterations=1)
    _count = 0
    _imgarr=[]
    thetas = []
    for item in _heart:

        #157=pi/2*100
        mask_max = 0
        mask_theta = 0
        # tmp_arr = []
        for i in range(24,290):
            black_img = np.zeros([_shape[0], _shape[1]], np.uint8)
            theta = float(i)*0.01
            y1 = int(item[1]-math.sin(theta)*item[2])
            x1 = int(item[0]+math.cos(theta)*item[2])
            # cv2.circle(black_img, (x1, y1), 2, 255, 3)
            # cv2.circle(black_img, (item[0], item[1]), 2, 255, 3)
            cv2.line(black_img, (item[0], item[1]), (x1, y1), 255, 5)
            tmp_img = cv2.bitwise_and(black_img, _img1)
            tmp = cv2.countNonZero(tmp_img)/cv2.countNonZero(black_img)
            # tmp_arr.append(tmp)
            if tmp>mask_max:
                mask_max=tmp
                mask_theta=theta
            # imwrite(dir_path+'/2_line1.jpg', black_img)

        if mask_max <0.28 and len(thetas)==1:

            for i in range(24, 290):
                black_img = np.zeros([_shape[0], _shape[1]], np.uint8)
                theta = float(i) * 0.01
                y1 = int(item[1] - math.sin(theta) * item[2])
                x1 = int(item[0] + math.cos(theta) * item[2])
                # cv2.circle(black_img, (x1, y1), 2, 255, 3)
                # cv2.circle(black_img, (item[0], item[1]), 2, 255, 3)
                cv2.line(black_img, (item[0], item[1]), (x1, y1), 255, 5)
                tmp_img = cv2.bitwise_and(black_img, org)
                tmp = cv2.countNonZero(tmp_img) / cv2.countNonZero(black_img)
                # tmp_arr.append(tmp)
                if tmp > mask_max:
                    mask_max = tmp
                    mask_theta = theta

        # from matplotlib.pyplot import plot, scatter, show
        # # a_ = [x for x in range(24,290)]
        # plot(tmp_arr)
        # # scatter(np.array(a_), np.array(tmp_arr), color='red')
        # show()
        black_img = np.zeros([_shape[0], _shape[1]], np.uint8)
        y1 = int(item[1] - math.sin(mask_theta) * item[2])
        x1 = int(item[0] + math.cos(mask_theta) * item[2])
        cv2.line(black_img, (item[0], item[1]), (x1, y1), 255, 8)
        imwrite(dir_path+'/3_theta'+str(_count)+'.jpg',black_img)
        print('img', dir_path+'/3_theta'+str(_count)+'.jpg')
        #
        # black_img1 = np.zeros([_shape[0], _shape[1]], np.uint8)
        # r = item[2]-20 if item[2]==_heart[1][2] else _heart[1][2]+ _heart[0][1]-_heart[1][1]-20
        # y1 = int(item[1] - math.sin(mask_theta) * (r))
        # x1 = int(item[0] + math.cos(mask_theta) * (r))
        # cv2.line(black_img1, (item[0], item[1]), (x1, y1), 255, 7)

        _img = cv2.subtract(_img,black_img)
        _img1 = cv2.subtract(_img1,black_img)
        imwrite(dir_path+'/3_theta__' + str(_count) + '.jpg', _img)
        print('img', dir_path+'/3_theta__' + str(_count) + '.jpg')
        _imgarr.append(_img)
        thetas.append(mask_theta)
        _count +=1

    imwrite(dir_path+'/3_non_pointer.jpg', _imgarr[1])
    print('img', dir_path+'/3_non_pointer.jpg')
    return _imgarr,thetas











def calcAngle(angles1):
    avgAngles = []
    angles = []
    for thet in angles1:
        if len(angles) > 0:
            ind = 0
            for ans in angles:
                ind += 1
                # print('thet',thet)
                # print(abs(thet -sum(ans)/len(ans)))
                if abs(thet - sum(ans) / len(ans)) < 1:
                    ans.append(thet)
                    # print(ans)
                else:
                    if ind == len(angles):
                        angles.append([thet])
                        break
                    else:
                        continue
        else:
            angles.append([thet])

    for ans in angles:
        avgAngles.append(sum(ans) / len(ans))




def convertPolar(zone,_heart,type,zone2=None):
    zoneshape = zone.shape
    zone = cv2.resize(zone, (zoneshape[1] * 8, zoneshape[0] * 8))
    # imwrite(dir_path+'/6_cut_numZoneCanny.jpg', zone)
    M=zoneshape[1] * 4/math.log(_heart[2]*4)
    # print(M)
    # 极坐标转换
    polar = cv2.logPolar(zone, (_heart[0] * 8, _heart[1] * 8), M, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if type==2:

        zone2 = cv2.resize(zone2, (zoneshape[1] * 8, zoneshape[0] * 8))
        polar2 = cv2.logPolar(zone2, (_heart[0] * 8, _heart[1] * 8), M, cv2.WARP_FILL_OUTLIERS)
        polar2 = cv2.rotate(polar2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        try:
            non_area, area, unused1 = findContours(cv2.dilate(polar, kernel5, iterations=1), polar, dst=polar2)
            # print(non_area)
        except Exception as err:
            print('err',err)
            print('line:416 in convertPolar(), please check image  '+dir_path+'/5_cut_res2.jpg')
            sys.exit(convertPolar_ERR)

    else:
        try:
            unused, area, unused1 = findContours(cv2.dilate(polar, kernel5, iterations=1), polar)
        except Exception as err:
            print('err',err)
            print('line:423 in convertPolar(), please check image  '+dir_path+'/5_cut_res1.jpg')
            sys.exit(convertPolar_ERR)

    imwrite(dir_path+'/6_cut_polar'+str(type)+'.jpg', polar)
    print('img', dir_path+'/6_cut_polar'+str(type)+'.jpg')


    numsShape = area.shape
    numsCanny = cv2.resize(area, (numsShape[1] * 4, numsShape[0] * 4))

    threshold, area = cv2.threshold(numsCanny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imwrite(dir_path+'/6_nums_kedu'+str(type)+'.jpg', area)  # 直着的 带刻度带数字图
    print('img', dir_path+'/6_nums_kedu'+str(type)+'.jpg')


    if type==1:

        border, ho = getLineBorder(area, 20)
        try:
            if len(border)<3:
                raise Exception('border less than 3 ')
        except Exception as err:
            print('err',err)
            print('in function convertPolar please check image '+dir_path+'/6_nums_kedu'+str(type)+'.jpg line :442')
            sys.exit(convertPolar_ERR)

        _kedu = area[border[2]:, :]
        _num = area[border[0]:border[1],:]
    else:
        non_numsCanny = cv2.resize(non_area, (numsShape[1] * 4, numsShape[0] * 4))
        threshold, non_area = cv2.threshold(non_numsCanny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border, ho = getLineBorder(non_area, 20)
        imwrite(dir_path+'/6_nums_kedu'+str(type)+'.jpg', non_area)  # 直着的 带刻度带数字图
        print('img', dir_path+'/6_nums_kedu'+str(type)+'.jpg')
        try:
            if len(border) < 3:
                raise Exception('border less than 3 ')
        except Exception as err:
            print('err',err)
            print('in function convertPolar please check image  '+dir_path+'/6_nums_kedu'+str(type)+'.jpg line: 456')
            sys.exit(convertPolar_ERR)

        _num = area[border[2]:border[3], :]
        _kedu = area[border[0]:, :].copy()
        _kedu[border[2]-border[0]:border[3]-border[0], :]=0

    imwrite(dir_path+'/6_nums_'+str(type)+'.jpg', _num)  # 直着的 带刻度带数字图
    print('img', dir_path+'/6_nums_'+str(type)+'.jpg')
    imwrite(dir_path+'/6_kedu_'+str(type)+'.jpg', _kedu)  # 直着的 带刻度带数字图
    print('img', dir_path+'/6_kedu_'+str(type)+'.jpg')
    return _kedu,_num



def convert2Num(onehot):
    '''

    :param onehot: type nparray only
    :return:
    '''
    # print(onehot)
    p= np.where(onehot==np.max(onehot))
    # print(p[1][0])
    # print('可能性：',onehot[p])
    return str(int(p[1][0]))


def processNum(cnn,numarea,k_rate,index):

    numsimg = numarea
    numsdilate = cv2.erode(numsimg, kernel4, iterations=2)
    numsdilate = cv2.dilate(numsdilate, kernel9, iterations=3)
    imwrite(dir_path+'/9_numsdilate'+str(index)+'.jpg', numsdilate)
    print('img', dir_path+'/9_numsdilate'+str(index)+'.jpg')
    numsArr = getEachNum(numsdilate)[0]
    numsimg = cv2.erode(numsimg, kernel8, iterations=1)
    nums = cutNums(numsArr, numsimg, dir_path+'/num'+str(index)+'/')
    _index = 0
    numbers_ = []
    for one_num in nums:
        num_border = getEachNum(one_num)[0]
        tmpArr = []

        maxWidth=0
        for i in range(0, len(num_border), 2):
            _width = num_border[i + 1]-num_border[i]
            if _width>maxWidth:
                maxWidth=_width

        for i in range(0, len(num_border), 2):

            new_cut = one_num[:, num_border[i]:num_border[i + 1]]
            blank = np.zeros((new_cut.shape[0],maxWidth),np.uint8)
            # print(new_cut.shape,blank.shape)
            blank[:,blank.shape[1]-new_cut.shape[1]:blank.shape[1]]=new_cut
            new_cut = cv2.resize(blank, (32, 64))
            imwrite(dir_path+'/num'+str(index)+'/' + str(_index) + '_' + str(int(i / 2)) + '.jpg', new_cut)

            # new_cut = Image.fromarray(cv2.cvtColor(new_cut, cv2.cv2.COLOR_BGR2GRAY))
            # hight, width = new_cut.shape
            # new_cut = np.asarray(new_cut)
            # new_cut = new_cut.reshape(1, hight * width)[0]
            tmpArr.append(new_cut)
        _index += 1
        numbers_.append(tmpArr)

    testDigits = [numbers_[0], numbers_[len(numbers_) - 1]]
    kedu_range = []
    for test in testDigits:
        res = ''
        for i in test:
            x = i.astype(float)
            x *= (1. / 255)
            x = x.reshape(1, 64, 32, 1)
            pr = cnn.predict(x)
            pr = convert2Num(pr)
            print('cnn识别结果：',pr)
        #     i = i.astype(float)
        #     i /= 255
        #     i = i.reshape(1,128*64)[0]
        #     predict = slearn.predict(i)
            res += '-' if str(pr) == '10' else str(pr)
        # print('softmax识别为:',res)
        try:
            kedu_range.append(int(res))
        except Exception as err:
            print('err',err)
            print('cnn 检测到'+res+' 示数检测有问题，请检查 ./v3/num'+str(index)+'文件夹中的数字')
            sys.exit(NUM_ERR)


    # result = k_pos * (k_len - 1) / (kedu_range[1] - kedu_range[0]) + kedu_range[0]
    result2 = k_rate*(kedu_range[1] - kedu_range[0])+ kedu_range[0]
    # print('结果：',result)
    print('res',result2)

    return result2




def processKedu(zone,index):

    ver = projectVertical(zone)
    (h1, w1) = zone.shape
    newHorizon = np.zeros([h1, w1], np.uint8)
    for i in range(0, w1):
        for j in range(0, ver[i]):
            newHorizon[j, i] = 255
    imwrite(dir_path+'/7_nums_kedu_'+str(index)+'.jpg', newHorizon)
    print('img', dir_path+'/7_nums_kedu_'+str(index)+'.jpg')
    maxtab, mintab = peakdetective.peakdet(ver, 3)

    # print(maxtab)
    k_res = list(maxtab[:, 1])

    # print(res.index(max(res)),len(res))
    k_pos = k_res.index(max(k_res))
    k_len = len(k_res)
    # print('指针位置',k_res.index(max(k_res)), '总刻度线个数',len(k_res))
    # print('position', k_res.index(max(k_res)) / (len(k_res) - 1) * 100)

    border1=maxtab[0][0]
    border2=maxtab[len(maxtab)-1][0]

    pos = maxtab[k_pos][0]

    rate = float(pos-border1)/(border2-border1)
    # print('比例计算：', rate)

    return k_pos,k_len,rate



def getScaleArea(heart_arr,_img,non_img_1,_non,theta_):
    '''

    :param heart_arr: 圆心数组
    :param _img: 表盘二值图
    :param non_img_1: 无指针图
    :param _non: 无指针图
    :param theta_: array 指针斜率
    :return:
    '''
    h,w = _img.shape
    hearts = heart_arr[:3]



    r0=int(hearts[2][2]+10)
    o0 = hearts[2]
    r1=int(hearts[2][1]-hearts[0][1])
    o1=hearts[2]
    r2 = int(hearts[2][2]-r1 + hearts[1][2])
    o2 = hearts[1]
    blank_img = np.zeros((h, w), np.uint8)
    blank_img0 = cv2.circle(blank_img,(o2[0],o2[1]),r2,255,-1)
    blank_img1 = cv2.circle(blank_img0, (o0[0], o0[1]), r0, 0, -1)
    blank_img = np.zeros((h, w), np.uint8)
    blank_img0 = cv2.circle(blank_img, (o0[0], o0[1]), r0, 255, -1)
    blank_img2 = cv2.circle(blank_img0, (o1[0], o1[1]), r1, 0, -1)

    _img1 = cv2.bitwise_and(blank_img1, _img)
    imwrite(dir_path+'/5_area1.jpg', _img1)
    print('img', dir_path+'/5_area1.jpg')

    _img2 = cv2.bitwise_and(blank_img2, non_img_1)
    imwrite(dir_path+'/5_area2.jpg', _img2)
    print('img', dir_path+'/5_area2.jpg')


    _non1 = cv2.bitwise_and(_non, blank_img1)
    _non2 = cv2.bitwise_and(_non, blank_img2)

    eroded1 = cv2.erode(_img1, kernel3, iterations=1)
    eroded1 = cv2.dilate(eroded1, kernel5, iterations=2)
    imwrite(dir_path+'/5_cut_dilate1.jpg', eroded1)
    print('img', dir_path+'/5_cut_dilate1.jpg')
    eroded2 = cv2.erode(_img2, kernel3, iterations=1)
    eroded2 = cv2.dilate(eroded2, kernel5, iterations=2)
    imwrite(dir_path+'/5_cut_dilate2.jpg', eroded2)
    print('img', dir_path+'/5_cut_dilate2.jpg')

    _hearts= hearts.copy()

    try:
        cut_non1, _cutnumZone1, hearts1 = findContours(eroded1, _img1, _hearts,_non1)
    except Exception as err:
        print('err',err)
        print('in function getScaleArea line 602, please check image  '+dir_path+'/5_cut_dilate1.jpg')
        sys.exit(getScaleArea_ERR)
    try:
        cut_non2, _cutnumZone2, hearts2 = findContours(eroded2, _img2, _hearts,_non2)

        point = hearts2[2][:2]
        _shape = _cutnumZone2.shape
        theta = theta_[1]
        y1 = int(point[1] - math.sin(theta) * _shape[0]*2)
        x1 = int(point[0] + math.cos(theta) * _shape[0]*2)
        # cv2.circle(black_img, (x1, y1), 2, 255, 3)
        # cv2.circle(black_img, (item[0], item[1]), 2, 255, 3)
        cv2.line(_cutnumZone2, (point[0], point[1]), (x1, y1), 255, 2)



    except Exception as err:
        print('err',err)
        print('in function getScaleArea line 609, please check image  '+dir_path+'/5_cut_dilate2.jpg')
        sys.exit(getScaleArea_ERR)
    # unused, non_numZone_adp,unused = findContours(eroded2, none_pointer_img)
    imwrite(dir_path+'/5_cut_res1.jpg', _cutnumZone1)
    print('img', dir_path+'/5_cut_res1.jpg')
    imwrite(dir_path+'/5_cut_res2.jpg', _cutnumZone2)
    print('img', dir_path+'/5_cut_res2.jpg')
    imwrite(dir_path+'/5_cut_p_img1.jpg', cut_non1)
    print('img', dir_path+'/5_cut_p_img1.jpg')
    imwrite(dir_path+'/5_cut_p_img2.jpg', cut_non2)
    print('img', dir_path+'/5_cut_p_img2.jpg')
    return _cutnumZone1,_cutnumZone2,hearts1[1],hearts2[2],cut_non1,cut_non2

def load_cnn():
    # model_path = './cnn/num_cnn.h5'
    model_path = './cnn/myCNN_pointer2.h5'
    K.clear_session()  # Clear previous models from memory.
    try:
        cnn_model = load_model(model_path)
    except:
        print('err','加载模型出错')
        sys.exit(MODEL_ERR)
    return cnn_model










def main(path,outPath = './result.txt'):


    #查找仪表圆形区域
    print('1.查找仪表圆形区域')
    # print(path)
    cut_Img,cut_origin,grayImg,gray_origin = findMainZone(path)
    # cut_Img,cut_origin,grayImg = findMainZone('./位置3/35/image2.jpg')
    # cut_Img,cut_origin,grayImg = findMainZone('./位置4/image1.jpg')
    # cut_Img,cut_canny,cut_origin,grayImg = findMainZone('./image12.jpg')

    # 找圆心
    print('2.找圆心')
    heartsArr = findHearts(gray_origin, cut_origin)

    #查找指针位置
    print('3.查找指针位置')
    # pointer_img = findPointer(cut_canny)
    # pointer_img = findPointer(cut_Img,heartsArr)
    # pointer_img = findPointer(gray2,heartsArr)
    #指针角度计算
    # calcAngle(ang1)
    non_img_arr,theta_arr = findPointer2(cut_Img, heartsArr[1:3])

    print('4.提前加载模型')
    cnn = load_cnn()

    print('5.裁剪刻度区域')
    cut_Img1,cut_Img2,heart1,heart2,non_img1,non_img2=getScaleArea(heartsArr, cut_Img,non_img_arr[0],non_img_arr[1],theta_arr)

    # print(heart1,heart2)
    print('6.进行极坐标转换')
    kedu1,num1 = convertPolar(cut_Img1, heart1,1)
    kedu2,num2 = convertPolar(cut_Img2, heart2,2,non_img2)
    #

    print('7.第一区域kedu处理')
    pos1,len1,rate1=processKedu(kedu1,index=1)

    print('8.第一区域数字处理')
    res1=  processNum( cnn,num1,rate1, index=1)

    print('9.第二区域kedu处理')
    pos2, len2, rate2= processKedu(kedu2,index=2)

    print('10.第二区域数字处理')
    res2 =processNum(cnn ,num2, rate2,index=2)

    f = open(outPath, 'a')

    f.write('\n'+path+' 结果1：'+str(res1)+ ' 结果2：'+str(res2))

    f.close()
    print('结果输出在'+outPath)

if __name__ == '__main__':
    args = sys.argv[1:]
    # print(args)
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--src', help='input image path')
    # parser.add_argument('--out', help='path of result txt ')
    # args =  parser.parse_args(args)
    # print(args)


    try:
        args.index('p1')
    except:
        p1 = 80
    else:
        p1 =int(args[args.index('p1') + 1])

    try:
        args.index('p2')
    except:
        p2 = 60
    else:
        p2 = int(args[args.index('p2') + 1])

    try:
        args.index('p3')
    except:
        p3 = 100
    else:
        p3 = int(args[args.index('p3') + 1])

    try:
        args.index('p4')
    except:
        p4 = 8
    else:
        p4 = int(args[args.index('p4') + 1])

    try:
        args.index('p5')
    except:
        p5 = 4
    else:
        p5 = int(args[args.index('p5') + 1])

    try:
        args.index('p6')
    except:
        p6 = 10
    else:
        p6 = int(args[args.index('p6') + 1])

    kernel8 = cv2.getStructuringElement(cv2.MORPH_RECT, (p5, p5))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (p6, p6))

    try:
        args.index('out')
    except:
        if len(args) == 0:
            print('请输入图片路径')
            sys.exit(0)
        elif len(args) >= 1:
            print('图片路径：', args[0])
            img_path = args[0]
            dir_path_arr =img_path.split(os.path.sep)[:-1]
            filename = img_path.split(os.path.sep)[-1].split('.')[0]
            dir_path = os.path.sep.join(dir_path_arr) + os.path.sep + filename
            isExists = os.path.exists(dir_path)
            if not isExists:
                os.makedirs(dir_path)
                os.makedirs(dir_path+'/num1')
                os.makedirs(dir_path+'/num2')
            main(img_path)
    else:
        out_path = args[args.index('out') + 1]
        print('图片路径：', args[0])
        print('结果保存路径：', out_path)
        img_path = args[0]
        dir_path_arr = img_path.split(os.path.sep)[:-1]
        filename = img_path.split(os.path.sep)[-1].split('.')[0]
        dir_path = os.path.sep.join(dir_path_arr)+os.path.sep+filename
        isExists = os.path.exists(dir_path)
        if not isExists:
            os.makedirs(dir_path)
            os.makedirs(dir_path + '/num1')
            os.makedirs(dir_path + '/num2')
        main(img_path, out_path)


    # if len(args) == 0:
    #     print('请输入图片路径')
    #     sys.exit(0)
    # elif len(args) == 1:
    #     print('图片路径：', args[0])
    #     img_path = args[0]
    #     main(img_path)
    # elif len(args) == 2:
    #     print('图片路径：', args[0])
    #     print('结果保存路径：', args[1])
    #     img_path = args[0]
    #     out_path = args[1]
    #     main(img_path, out_path)





    # bl = np.zeros((300,500),np.uint8)
    # try:
    #     int('1-')
    # except Exception as err:
    #     print('err',err)