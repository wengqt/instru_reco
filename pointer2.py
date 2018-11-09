import cv2
import numpy as np
import sys
import os
import math
import peakdetective
from keras.models import load_model

from keras import backend as K

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernel6 = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 30))
kernel8 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 25))
kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))


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
    cv2.imwrite('./pointer2/cut_findContours.jpg', new_img_canny)
    sets = [x-offset1,y-offset1, w+offset2+offset1, h+offset2+offset1]
    _heart= None
    if heartsarr is not None:
        _heart = heartsarr.copy()
        # print(heartsarr)
        for a in _heart:
            a[0] =a[0]-x+offset1
            a[1] = a[1] - y + offset1
    return new_img,new_img_canny,_heart,sets






def findMainZone(path):
    # print(path)
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
    # cv2.imwrite(dir_path+'/canny.jpg', canny)

    canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    # canny = cv2.cvtColor(cv2.bilateralFilter(img1, 9, 70, 70), cv2.COLOR_BGR2GRAY)
    '''
    找到仪表区域
    '''
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=60, minRadius=50)
    circles = np.uint16(np.around(circles))
    img1_cp = img1.copy()
    for i in circles[0, 0:3]:
        # 画圆圈
        cv2.circle(img1_cp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img1_cp, (i[0], i[1]), 2, (0, 0, 255), 3)

    # _circles = circles[0,:3] #投票数最大的圆
    # _circles = sorted(_circles,key=lambda x:x[2],reverse=True )
    # print(_circles)
    the_circle = circles[0][0]
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), the_circle[2], (0, 0, 255), 2)
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), 2, (0, 0, 255), 3)
    cv2.imwrite('./pointer2/1_circles.jpg', img1_cp)
    print('img', './pointer2/1_circles.jpg')
    # print(the_circle[1],the_circle[2])
    # print(int(the_circle[1]) - int(the_circle[2]))
    d1 =0 if int(the_circle[1]) - int(the_circle[2])<0 else int(the_circle[1]) - int(the_circle[2])
    d2 =0 if int(the_circle[0]) - int(the_circle[2])<0 else int(the_circle[0]) - int(the_circle[2])
    cutImg_o = img1[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    gray_o = canny[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    cv2.imwrite('./pointer2/1_cut.jpg', cutImg_o)
    print('img', './pointer2/1_cut.jpg')

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    # cut_g = cv2.GaussianBlur(cutImg, (7, 7), 1)
    # cut_g = cv2.equalizeHist(cut_g)
    # cutCanny = cv2.Canny(cut_g, 50, 100)
    # cv2.imwrite(dir_path+'/1_cut_Canny.jpg', cutCanny)
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg1 = cv2.bilateralFilter(cutImg, 9, 70, 70)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),0) #灰度图，用于后面的圆圈识别
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别

    cutImg3 = cv2.adaptiveThreshold(cutImg1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    # ret2, cutImg3 = cv2.threshold(cutImg2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('./pointer2/1_cut_adapt.jpg', cutImg3)
    print('img', './pointer2/1_cut_adapt.jpg')
    return cutImg3,cutImg_o



def diff_circle(src):
    _shape = src.shape
    z1 = cv2.countNonZero(src[0:30,0:30])
    z2 = cv2.countNonZero(src[_shape[0]-30:,0:30])
    z3 = cv2.countNonZero(src[0:30,_shape[1]-30:])
    z4 = cv2.countNonZero(src[_shape[0]-30:,_shape[1]-30:])
    if z1+z4>z2+z3:
        return 0
    else:
        return 1

def fitCircle2(cut,black,point_arr):
    type = diff_circle(cut)
    x1,y1 = point_arr[0]
    x2,y2 = point_arr[-1]
    xs=[]
    ys=[]
    for i in range(1,len(point_arr)-1):
        x0,y0=point_arr[i]
        k1 = float(y1 - y0) / (x1 - x0)
        k2 = float(y2 - y0) / (x2 - x0)
        k_3 = -1. / k1
        k_4 = -1. / k2
        if k_4==k_3:
            continue
        xm1 = (x1 + x0) / 2
        ym1 = (y1 + y0) / 2
        xm2 = (x2 + x0) / 2
        ym2 = (y2 + y0) / 2
        x_h =(k_3*xm1-k_4*xm2+ym2-ym1)/(k_3-k_4)
        y_h = k_4*(x_h-xm2)+ym2
        xs.append(x_h)
        ys.append(y_h)
        # r_h = math.sqrt((pow(x0-x_h,2)+pow(y0-y_h,2)))
    x_res = int(np.mean(xs))
    y_res = int(np.mean(ys))
    r_res =  int(math.sqrt((pow(x1-x_res,2)+pow(y1-y_res,2))))

    black_cp = black.copy()
    cv2.circle(black_cp,(x_res,y_res),r_res,255,2)
    cv2.imwrite('./pointer2/2_cut_find_heart.jpg', black_cp)
    return x_res, y_res, r_res, type

def fitCircle(src,zero_img,sets):
    '''
    左圆形为1，右圆形为0
    :param src:
    :param zero_img:
    :param sets:
    :return:
    '''
    [x,y,w,h] = sets
    type = diff_circle(src)
    if type == 0:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
    elif type == 1:
        x1 = x + w
        y1 = y
        x2 = x
        y2 = y + h

    xm = int((x1 + x2) / 2)
    ym = int((y1 + y2) / 2)

    k1 = float(y1 - y2) / (x1 - x2)

    k2 = -1. / k1
    mask_max = 0
    x0_max = xm
    y0_max = ym
    r_max = w
    alp = 1 if type==1 else -1
    for r in range(w, zero_img.shape[0], 5):
        x0 = ((ym + alp * k2 * ((k2 ** 2) * (r ** 2) - (k2 ** 2) * (x1 ** 2) + 2 * (k2 ** 2) * x1 * xm - (k2 ** 2) * (
                    xm ** 2) + 2 * k2 * x1 * y1 - 2 * k2 * x1 * ym - 2 * k2 * xm * y1 + 2 * k2 * xm * ym + (r ** 2) - (
                                      y1 ** 2) + 2 * y1 * ym - (ym ** 2)) ** (1 / 2) + k2 * x1 - k2 * xm + (
                           k2 ** 2) * y1) / ((k2 ** 2) + 1) - ym + k2 * xm) / k2
        y0 = k2 * (x0 - xm) + ym
        # print(x0,y0)
        black_img = np.zeros(zero_img.shape, np.uint8)
        try:
            cv2.circle(black_img, (int(x0), int(y0)), r, 255, 20)
        except Exception as  err:
            print('err',err)
            r +=20

        tmp = np.mean(cv2.bitwise_and(black_img, zero_img))
        if tmp > mask_max:
            mask_max = tmp
            x0_max = x0
            y0_max = y0
            r_max = r

    cv2.circle(zero_img, (int(x0_max), int(y0_max)), r_max, 255, 20)
    cv2.imwrite('./pointer2/2_cut_find_heart.jpg', zero_img)
    return x0_max,y0_max,r_max, type


def make_mask(src,type, offset1=0,offset2=-30,big_index=0):
    _, contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        raise Exception('最大区域矩形检测失败！', 'line 42 in function findContours')


    c = sorted(contours, key=cv2.contourArea, reverse=True)[big_index]
    zero = np.zeros(src.shape,np.uint8)
    rect = cv2.minAreaRect(c)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box1=[]
    box2=[]
    if type==1:
        offset1 = 0-offset1
    elif type==0:
        offset1 = offset1

    for [_x,_y] in box:
        tmp = _x+offset1, _y-offset2
        box1.append(tmp)

    for [_x,_y] in box:
        tmp = _x, _y-abs(int(offset1))
        box2.append(tmp)

    box1 = np.array(box1)
    box2 = np.array(box2)
    # print(box,box1)

    cv2.drawContours(zero, [box1], 0, 255, -1)
    cv2.drawContours(zero, [box2], 0, 255, -1)
    cv2.drawContours(zero, [box], 0, 255, -1)
    zero = cv2.dilate(zero, kernel9, iterations=3)
    cv2.imwrite('./pointer2/2_mask'+str(big_index)+'.jpg',zero)
    return zero



def findCircleContours(mask,big_index=0,points_num=10):

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise Exception('最大区域矩形检测失败！', 'line 42 in function findContours')

    c = sorted(contours, key=cv2.contourArea, reverse=True)[big_index]
    x, y, w, h = cv2.boundingRect(c)
    c1 = np.reshape(c, (-1, 2))
    points_arr_index = [int(len(c1)/(2*points_num))*i for i in range(points_num) ]
    # print(points_arr_index)
    c_y=c1[:,1]
    c_x=c1[:,0]
    points_arr_y = [c_y[i] for i in points_arr_index]
    points_arr_x=[]
    for p in points_arr_y:
        tmp=[]
        ind =  np.where(c_y == p)[0]
        # print(ind)
        if len(ind)==0:
            del points_arr_y[points_arr_y.index(p)]
        else:
            for i in ind:
                tmp.append(c_x[i])
                # print(c_x[i])
            points_arr_x.append(int(np.mean(tmp)))

    # print(points_arr_x,points_arr_y)
    points_arr=list(zip(points_arr_x,points_arr_y))
    black = np.zeros(mask.shape,np.uint8)
    cv2.drawContours(black,[c],0,255,-1)
    # for i in points_arr:
    #     cv2.circle(black,tuple(i),10,255,-1 )

    cv2.imwrite('./pointer2/2_mask_draw.jpg',black)
    cut = black[y:y+h,x:x+w]
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # smallBox = np.int0(box)
    return black,cut,points_arr






def findHearts(src1,org):
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
    src = src1.copy()
    src = cv2.erode(src, kernel3)
    src = cv2.dilate(src,kernel7)
    src = cv2.erode(src, kernel8,iterations=2)

    cv2.imwrite('./pointer2/2_erode.jpg', src)

    areas = []
    type_arr = []
    for i in range(0,2):

        # zeros,cut_img,points_arr = findCircleContours(src,big_index=i,points_num=20)
        # x0_max, y0_max, r_max, _type = fitCircle2(cut_img, zeros, points_arr)

        # 求圆心
        un,cut_img,un,_sets = findContours(src,src,dst=src1,big_index=i)
        [x, y, w, h] = _sets
        zeros = np.zeros(src.shape, np.uint8)
        zeros[y:y+h,x:x+w] = cut_img
        x0_max, y0_max, r_max, _type = fitCircle(cut_img, zeros, _sets)

        of1 = int(r_max/4) if _type==1 else int(r_max/3.5)
        #制作分割蒙版
        mask = make_mask(src,_type,offset1=int(r_max/4),big_index=i)
        mask_zeros = cv2.bitwise_and(src1, mask)
        if _type==0:
            rows, cols = mask_zeros.shape
            M = cv2.getRotationMatrix2D((x0_max, y0_max), 20, 1.0)
            mask_zeros = cv2.warpAffine(mask_zeros, M, (cols, rows))


        cv2.imwrite('./pointer2/2_zeros'+str(i)+'.jpg', mask_zeros)
        #极坐标转换
        zoneshape = mask_zeros.shape
        zone = cv2.resize(mask_zeros, (zoneshape[1]*4, zoneshape[0]*4 ))
        # cv2.imwrite('./v3/6_cut_numZoneCanny.jpg', zone)
        M = zoneshape[1]*3/ math.log(r_max*3)
        polar = cv2.logPolar(zone, (x0_max*4, y0_max*4), M, cv2.WARP_FILL_OUTLIERS)
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite('./pointer2/2_polar'+str(i)+'.jpg', polar)
        areas.append(polar)
        type_arr.append(_type)
    # print(type_arr)
    return areas,type_arr


# def skeleton(img_adp):
#     size = np.size(img_adp)
#     ps = cv2.countNonZero(img_adp)
#     skel = np.zeros(img_adp.shape, np.uint8)
#
#     # ret, img = cv2.threshold(img, 127, 255, 0)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 2))
#     done = False
#
#     while (not done):
#         eroded = cv2.erode(img_adp, element)
#         temp = cv2.dilate(eroded, element)
#         temp = cv2.subtract(img_adp, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         img_adp = eroded.copy()
#
#         rate = float(ps)/cv2.countNonZero(img_adp)
#         if rate >= 1.5:
#             done = True
#
#     cv2.imwrite('./pointer2/3_skeleton.jpg',img_adp)
#     return img_adp


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


def getLineBorder(src,min_range=0,d2=10):
    (h1, w1) = src.shape
    horizon = [0 for z in range(0, h1)]

    for i in range(0, h1):  # 遍历一行
        for j in range(0, w1):  # 遍历一列
            if src[i, j] != 0:
                horizon[i] += 1

    # line_border = []

    border_arr = get_border(horizon, min_range,d2)

    return border_arr,horizon


def projectVertical(src):
    (h1, w1) = src.shape
    vertical = [0 for z in range(0, w1)]

    for i in range(0, w1):  # 遍历一lie
        for j in range(0, h1):  # 遍历一hang
            if src[j, i] != 0:
                vertical[i] += 1
    return vertical




def black_img(src):
    black = np.zeros((src.shape[0]+30,src.shape[1]),np.uint8)
    black[15:src.shape[0]+15,:] = src
    return black

def split_area(src_arr):
    index=0
    _areas=[]
    for src in src_arr:
        src1 = src.copy()
        src1 = cv2.dilate(src1,kernel8,iterations=2)
        un,area,un,un = findContours(src1,src)
        area = cv2.erode(area,kernel3)
        area = cv2.dilate(area,kernel4)
        ret, area = cv2.threshold(area, 127, 255, 0)
        lines,un = getLineBorder(area,20,20)
        print(lines)
        # for i in lines:
        #     area[i,:]=255

        cv2.imwrite('./pointer2/3_area' + str(index) + '.jpg', area)
        _areas.append({'nums': black_img(area[lines[0]:lines[1], :]),
                       'kedu1':black_img(area[lines[2]:lines[3], :]) ,
                       'kedu2':black_img(area[lines[2]:lines[4], :])})
        index+=1

    return _areas


def angle_corract(src,src2):
    # canny = cv2.Canny(src, 0, 50)
    # canny = cv2.dilate(canny,kernel3)
    lines = cv2.HoughLines(src, 1, np.pi / 180, 40)
    (height, width) = src.shape[:2]
    numLine = 0
    angle = 0
    # lines1 = lines[:,0,:]
    # for rho, theta in lines1[:]:
    #     if abs(theta) < np.pi / 6:
    #         print('a angle is :', theta * 180 / np.pi, 'and ', (np.pi / 2 - theta) * 180 / np.pi)
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x11 = int(x0 + 1000 * (-b))
    #         y11 = int(y0 + 1000 * (a))
    #         x22 = int(x0 - 1000 * (-b))
    #         y22 = int(y0 - 1000 * (a))
    #         cv2.line(src, (x11, y11), (x22, y22), 255, 1)
    # cv2.imwrite('./pointer2/line.jpg', src)
    for ls in lines:
        for line in ls:
            rho = line[0]
            theta = line[1]
            if abs(np.pi / 2 - theta) < np.pi / 6:
                # print('this angle : ', theta * 180 / np.pi)
                numLine = numLine + 1
                angle = angle + theta

            # if angle > (np.pi / 2):
            #     angle = angle - np.pi

    averageAngle = (angle / float(numLine)) * 180 / np.pi
    print('averageAngle : %f' % averageAngle)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), averageAngle - 90, 1.0)

    correct = cv2.warpAffine(src, M, (width, height))
    correct2 = cv2.warpAffine(src2, M, (width, height))
    cv2.imwrite('./pointer2/correct.jpg', correct)
    cv2.imwrite('./pointer2/correct2.jpg', correct2)

    return correct,correct2





def processKedu(arr,type_arr):
    # print(len(arr),type_arr)
    rate_arr=[]
    index_arr=[]
    for ind in range(len(arr)):
        kedu1,kedu2= angle_corract(arr[ind]['kedu1'],arr[ind]['kedu2'])
        ver = projectVertical(kedu1)
        ver2 = projectVertical(cv2.subtract(kedu2,kedu1))
        pos = np.mean(np.where(ver2 == np.max(ver2)))
        print(pos)
        (h1, w1) = kedu1.shape
        newblack = np.zeros([h1, w1], np.uint8)
        for i in range(0, w1):
            for j in range(0, ver[i]):
                newblack[j, i] = 255
        cv2.imwrite('./pointer2/4_nums_kedu_' + str(1) + '.jpg', newblack)

        maxtab,mintab = peakdetective.peakdet(ver,5)
        maxs = maxtab[:,1]
        tabs = maxtab[:,0]

        max_num = np.max(maxs)
        max_index = np.where((maxs<max_num+10)&(maxs>max_num-10))[0]
        tops=[]
        for m in max_index:
            tops.append(tabs[m])

        _left=0
        _right=0
        pos_index = 0

        # print(i)
        if type_arr[ind]==0:
            tops.reverse()
            tops.append(tabs[0])
            for t in range(len(tops)):
                if tops[t] <=pos:
                    _left=tops[t-1]
                    _right=tops[t]
                    pos_index=t-1
                    break

        else:
            tops.append(tabs[-1])
            for t in range(len(tops)):
                if tops[t] >=pos:
                    _left=tops[t-1]
                    _right=tops[t]
                    pos_index=t-1
                    break
        print('节点数', len(tops), tops)
        rate = (pos-_left)/(_right-_left)
        print(rate,pos_index)
        rate_arr.append(rate)
        index_arr.append(pos_index)
    return rate_arr,index_arr



def processNum(arr,type_arr):

    num_arr=[]
    for i in range(2):

        num_mask = cv2.dilate(arr[i]['nums'],kernel6)
        cv2.imwrite('./pointer2/5_num_dilate.jpg',num_mask)
        _, contours, hierarchy = cv2.findContours(num_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise Exception('最大区域矩形检测失败！', 'line 42 in function findContours')
        tmp_contours=[]
        for c in contours:
            rect = cv2.boundingRect(c)
            tmp_contours.append(rect)

        rects = sorted(tmp_contours, key=lambda x: (x[0]))
        nums=[]
        index =0
        # print(rects)
        for x, y, w, h in rects:
            num_box = arr[i]['nums'][y:y+h,x:x+w]
            # _, m_contours, hierarchy = cv2.findContours(num_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # angles=[]
            index+=1
            # cv2.imwrite('./pointer2/5_num_'+str(index)+'.jpg', num_box)
            get_number(num_box,type_arr[i],index)
            nums.append(num_box)
            # for mc in m_contours:
            #     rect = cv2.minAreaRect(mc)
            #     box = cv2.boxPoints(rect)
            #     smallBox = np.int0(box)
            #     print(smallBox)
            #     xs=smallBox[:,0].tolist()
            #     ys=smallBox[:,1].tolist()
            #     x1_index = xs.index(max(xs))
            #     y1_index =ys.index(min(ys))
            #     if smallBox[y1_index][0]-smallBox[x1_index][0] !=0 and smallBox[y1_index][1]-smallBox[x1_index][1] !=0:
            #         k = (smallBox[y1_index][1]-smallBox[x1_index][1])/(smallBox[y1_index][0]-smallBox[x1_index][0])
            #         thet = math.atan(k) * 180 / np.pi
            #         print('倾斜角',thet)
            #         angles.append(thet)
            #     # cv2.drawContours(num_box, [smallBox], 0, 255, 1)
            # print(angles)
            # avg_angle = np.mean(np.array(angles))
            #
            # if type_arr[i]==0:
            #     avg_angle =  avg_angle - 90
            # points = np.where(num_box==255)
            # points = np.array(list(zip(points[0],points[1])))
            # print(list(points))
            # height, width = num_box.shape
            # bl = np.zeros((height,width),np.uint8)
            # for it in points:
            #     bl[it[0]][it[1]]=255
            # cv2.imwrite('./pointer2/5_num_dilate.jpg', bl)
            # m_contours = sorted(m_contours, key=cv2.contourArea, reverse=True)

            # height, width = num_box.shape
            # for points in m_contours:

                # (x, y), (MA, ma), angle = cv2.fitEllipse(points)
                # y1 = int(y - math.sin(angle) * 1000)
                # x1 = int(x + math.cos(angle) *1000)
                # ellipse = cv2.fitEllipse(points)
                # cv2.line(num_box, (int(x), int(y)), (x1, y1), 255, 1)
                # cv2.ellipse(num_box, ellipse, 255, 1)
                # angles.append(angle)

                # [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 1e-2, 1e-2)
                # lefty = int((-x * vy / vx) + y)
                # righty = int(((width - x) * vy / vx) + y)
                # thet = math.atan(vy / vx) * 180 / np.pi
                # angles.append(thet)
                # cv2.line(num_box, (width - 1, righty), (0, lefty), 255, 1)
            #     matchNum(points)
            #
            # cv2.imwrite('./pointer2/5_num_box.jpg', num_box)
            # avg_angle = np.mean(np.array(angles))
            # if type_arr[i] == 1:
            #     avg_angle =  avg_angle + 90
            # else:
            #     avg_angle =  avg_angle -90
            #
            # print(avg_angle)
            # M = cv2.getRotationMatrix2D((width / 2, height / 2), avg_angle, 1.0)
            # heightNew = int(width * math.fabs(math.sin(math.radians(avg_angle))) + height * math.fabs(math.cos(math.radians(avg_angle))))
            # widthNew = int(height * math.fabs(math.sin(math.radians(avg_angle))) + width * math.fabs(math.cos(math.radians(avg_angle))))
            # M[0, 2] += (widthNew - width) / 2
            # M[1, 2] += (heightNew - height) / 2
            # num_box = cv2.warpAffine(num_box, M, (widthNew, heightNew))
            # cv2.imwrite('./pointer2/5_num_box.jpg', num_box)
            # _, t_contours, hierarchy = cv2.findContours(num_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # _index=0
            # for t in t_contours:
            #     x_, y_, w_, h_ = cv2.boundingRect(t)
            #     single_num = num_box[y_:y_ + h_, x_:x_ + w_]
            #     _index+=1
            #     single_num = cv2.resize(single_num,(32, 64))
            #     cv2.imwrite('./pointer2/5_num_' + str(_index) + '.jpg', single_num)
            #     x = single_num.astype(float)
            #     x *= (1. / 255)
            #     x = x.reshape(1, 64, 32, 1)
            #     pr = cnn.predict(x)
            #     pr = convert2Num(pr)
            #
            #     print('cnn识别结果：', pr)
            # nums.append(num_box)
            # cv2.imwrite('./pointer2/5_num_box.jpg', num_box)
        if type_arr[i]==0:
            nums.reverse()
        num_arr.append(nums)
    return num_arr

def rotateImg(src,avg_angle,ind):
    height,width= src.shape
    M = cv2.getRotationMatrix2D((width / 2, height / 2), avg_angle, 1.0)
    heightNew = int(width * math.fabs(math.sin(math.radians(avg_angle))) + height * math.fabs(math.cos(math.radians(avg_angle))))
    widthNew = int(height * math.fabs(math.sin(math.radians(avg_angle))) + width * math.fabs(math.cos(math.radians(avg_angle))))
    M[0, 2] += (widthNew - width) / 2
    M[1, 2] += (heightNew - height) / 2
    num_box = cv2.warpAffine(src, M, (widthNew, heightNew))
    cv2.imwrite('./pointer2/5_num_box_rotate' + str(ind) + '.jpg', num_box)
    return num_box


def get_number(num_img,type,index):
    _, m_contours, hierarchy = cv2.findContours(num_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if type==1:
    c = sorted(m_contours,key=lambda x:cv2.boundingRect(x)[1],reverse=True)
    # else:
    #     c = sorted(m_contours, key=lambda x: cv2.boundingRect(x)[1])



    rect = cv2.minAreaRect(c[0])
    box = cv2.boxPoints(rect)
    smallBox = np.int0(box)

    xs=smallBox[:,0].tolist()
    ys=smallBox[:,1].tolist()
    thet=0
    x1_index = xs.index(max(xs))
    y1_index =ys.index(min(ys))
    # cv2.drawContours(num_img, [smallBox], 0, 255, 1)
    if smallBox[y1_index][0]-smallBox[x1_index][0] !=0 and smallBox[y1_index][1]-smallBox[x1_index][1] !=0:
        k = (smallBox[y1_index][1]-smallBox[x1_index][1])/(smallBox[y1_index][0]-smallBox[x1_index][0])
        thet = math.atan(k) * 180 / np.pi

    elif smallBox[y1_index][0]-smallBox[x1_index][0] ==0 or smallBox[y1_index][1]-smallBox[x1_index][1]==0:
        thet=90
        if type==0:
            thet=0

    # print(smallBox[y1_index][0]-smallBox[x1_index][0],smallBox[y1_index][1]-smallBox[x1_index][1])
    if type == 0:
        thet = thet - 90
    print('倾斜角', thet)
    num_img = rotateImg(num_img,thet,index)

    _, _contours, hierarchy = cv2.findContours(num_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _contours = sorted(_contours,key=lambda x:cv2.boundingRect(x)[0])

    for m_c in _contours:
        # matchNum(m_c)
        x_, y_, w_, h_ = cv2.boundingRect(m_c)
        if w_>10 and h_>10:
            single_num = num_img[y_:y_ + h_, x_:x_ + w_]
            single_num = cv2.dilate(single_num,kernel4)
            if w_>=h_:
                single_num = cv2.resize(single_num,(30,int(h_*30/w_)))
                if int(h_*30/w_)>64:
                    single_num = cv2.resize(single_num, (30, 60))
            else:
                # single_num = cv2.resize(single_num, (int(w_ * 60/ h_),60))
                # if int(w_ * 60/ h_)>32:
                single_num = cv2.resize(single_num, (30, 60))


            hei,wid = single_num.shape
            black = np.zeros((64, 32), np.uint8)
            h_diff = int((64-hei)/2)
            w_diff = int((32-wid)/2)
            black[h_diff:hei+h_diff,w_diff:w_diff+wid]=single_num
            # black[0:hei,0:wid]=single_num

            cv2.imwrite('./pointer2/5_num_' + str(index) + '.jpg', black)
            x = black.astype(float)
            x *= (1. / 255)
            x = x.reshape(1, 64, 32, 1)
            pr = cnn.predict(x)
            pr = convert2Num(pr)

            print('cnn识别结果：', pr)


    # if type==0:
    #     res = [10, 30, 50, 70, 90]
    # elif type==1:
    #     res = [-10, 0, 10, 20, 30, 40]

    # return res[num_img]




def precessRes(nums,rate_arr,res_index,type_arr):
    for i in range(len(nums)):
        if type_arr[i]==0:
            pos = int(res_index[i]/2)
        elif type_arr[i]==1:
            pos = res_index[i]
        print('pos',pos)
        # _left = get_number(nums[i][pos],type_arr[i])
        _left = get_number(pos, type_arr[i])
        if len(nums[i]) == pos + 1:
            # _right = get_border(nums[i][pos + 1],type_arr[i])
            _right = get_number(pos-1, type_arr[i])
            arange = (_left-_right)/2
        else:
            # _right = get_border(nums[i][pos + 1],type_arr[i])
            _right = get_number(pos + 1, type_arr[i])
            arange = _right-_left

        res= arange*rate_arr[i]+_left
        print('结果',res)





def matchNum(src):
    ret_arr = []
    for i in range(0,11):
        img = cv2.imread('./cnn/imgs/'+str(i)+'.jpg',0)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, 2)
        cnt = contours[0]
        ret = cv2.matchShapes(cnt, src, 1, 0.0)
        # print(str(i),ret)
        ret_arr.append(ret)
    res = ret_arr.index(min(ret_arr))
    print('match结果:',res)





def load_cnn():
    model_path = './cnn/num_cnn.h5'
    K.clear_session()  # Clear previous models from memory.
    try:
        cnn_model = load_model(model_path)
    except:
        print('加载模型出错')
        # sys.exit(MODEL_ERR)
    return cnn_model

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



if __name__ == '__main__':
    # path = './位置2/30/image3.jpg'
    # path = './位置4/image3.jpg'
    path = './img/image2.jpg'

    # dir_path_arr = path.split(os.path.sep)[:-1]
    # filename = path.split(os.path.sep)[-1].split('.')[0]
    # dir_path = os.path.sep.join(dir_path_arr) + os.path.sep + filename

    cut_Img, cut_origin = findMainZone(path)

    print('2.找区域')
    area_arr,_typearr = findHearts(cut_Img, cut_origin)
    print('3.分割数字和刻度')
    split_arr =  split_area(area_arr)
    print('4.处理刻度')
    rates,num_index = processKedu(split_arr,_typearr)
    cnn = load_cnn()
    print('5.处理数字')
    number_arr = processNum(split_arr,_typearr)
    print('6.计算结果')
    # precessRes(number_arr,rates,num_index,_typearr)