import cv2
import numpy as np
import sys
import math
import peakdetective
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernel6 = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
kernel8 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
cross1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

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

    img1_cp = img1.copy()
    img1shape = img1.shape
    img1_cp = cv2.resize(img1_cp, (int(img1shape[1]/2), int(img1shape[0]/2)))


    # canny = cv2.Canny(cv2.GaussianBlur(img1, (7, 7), 0), 100, 250)
    # cv2.imwrite(dir_path+'/canny.jpg', canny)

    canny = cv2.cvtColor(cv2.GaussianBlur(img1_cp, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    # canny = cv2.cvtColor(cv2.bilateralFilter(img1, 9, 70, 70), cv2.COLOR_BGR2GRAY)
    '''
    找到仪表区域
    '''
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=60, minRadius=50)
    circles = np.uint16(np.around(circles))

    # print(circles)
    for i in circles[0,:]:
        # 画圆圈
        cv2.circle(img1_cp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img1_cp, (i[0], i[1]), 2, (0, 0, 255), 3)

    the_circle = circles[0][0]
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), the_circle[2], (0, 0, 255), 2)
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), 2, (0, 0, 255), 3)
    cv2.imwrite('./pointer3/1_circles.jpg', img1_cp)
    print('img', './pointer3/1_circles.jpg')

    the_circle = the_circle*np.array([2])
    # print(the_circle[1],the_circle[2])
    # print(int(the_circle[1]) - int(the_circle[2]))
    d1 =0 if int(the_circle[1]) - int(the_circle[2])<0 else int(the_circle[1]) - int(the_circle[2])
    d2 =0 if int(the_circle[0]) - int(the_circle[2])<0 else int(the_circle[0]) - int(the_circle[2])


    img2 = calc_gamma(img1)
    # img1 = calc_equalize(img1)
    # img2 = calc_lap(img2)
    #对裁剪的图片进行二值化和平滑处理。

    cutImg_o = img1[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    # testfillflood(cutImg_o)
    # test_gray(cutImg_o)
    # gray_o = canny[d1:the_circle[1] + the_circle[2],
    #            d2:the_circle[0] + the_circle[2]]
    cv2.imwrite('./pointer3/1_cut.jpg', cutImg_o)
    print('img', './pointer3/1_cut.jpg')

    cutImg = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)


    avg,stdd = bright_avg(cutImg)

    # cut_g = cv2.GaussianBlur(cutImg, (7, 7), 1)
    # cutImg = cv2.equalizeHist(cutImg)
    # cutCanny = cv2.Canny(cut_g, 50, 100)
    # cv2.imwrite(dir_path+'/1_cut_Canny.jpg', cutCanny)

    if avg<40:#暗光
        # cutImg = cv2.equalizeHist(cutImg)
        # dark = True
        pass
    else:
        light_mask = cv2.inRange(cutImg,250,255)
        img1 = contrast_brightness_image(img1, 0.4, 0,light_mask)
        cutImg = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        # dark = False


    # kernel = np.ones((5, 5), np.float32) / 25
    # cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg1 = cv2.bilateralFilter(cutImg, 5, 50, 50)
    # cutImg1 = cv2.equalizeHist(cutImg1)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),1) #灰度图，用于后面的圆圈识别
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别

    cutImg3 = cv2.adaptiveThreshold(cutImg1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cutImg3 = cv2.erode(cutImg3,cross1,iterations=1)
    # ret2, cutImg3 = cv2.threshold(cutImg1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cutImg3 = cv2.erode(cutImg3,kernel3)
    cutImg3 = cutImg3[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    cv2.imwrite('./pointer3/1_cut_adapt.jpg', cutImg3)
    print('img', './pointer3/1_cut_adapt.jpg')
    return cutImg3,cutImg_o,(the_circle[0]-d2,the_circle[1]-d1)



# def findCircleContours(src,big_index=0,points_num=20):
#     _, contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#

def bright_avg(src):
    avg,stdd = cv2.meanStdDev(src)
    print(avg)
    return avg[0][0],stdd[0][0]


def contrast_brightness_image(src1, a, g,mask=None):
    # h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    # src2 = np.zeros([h, w, ch], src1.dtype)
    if mask is None:
        h, w, ch = src1.shape
        src2 = np.zeros([h, w, ch], src1.dtype)
        b=1-a
    else:
        src2 = cv2.cvtColor(cv2.bitwise_not(mask),cv2.COLOR_GRAY2BGR)
        b=0.1
    cv2.imwrite('./pointer3/0_light_mask.jpg', src2)
    dst = cv2.addWeighted(src1, a, src2,b, g)  # addWeighted函数说明如下
    cv2.imwrite('./pointer3/0_weight.jpg', dst)
    return dst


def findHearts1(src1,heart1):
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
    # src = cv2.erode(src, kernel3)

    # src = cv2.dilate(src,kernel7)
    #
    # src = cv2.erode(src, kernel8,iterations=1)
    src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel5)
    cv2.imwrite('./pointer3/2_erode.jpg', src)
    # cv2.imwrite('./pointer3/2_erode.jpg', src)

    src_shape = src.shape

    fit_r = 0
    max_mean = 0
    for r in range(int(src_shape[0]/4.),int(src_shape[0]/2.5)):
        black = np.zeros(src_shape,np.uint8)
        cv2.circle(black,heart1,r,255,10)
        tmp = cv2.countNonZero(cv2.bitwise_and(black, src))
        if tmp>max_mean:
            max_mean = tmp
            fit_r = r

    cv2.circle(src, heart1, fit_r, 255, 10)
    cv2.imwrite('./pointer3/2_fit_circle.jpg', src)



    return [heart1[0],heart1[1],fit_r]


def findHearts2(org):
    if min(bright_avg(org))>110:
        pass
    else:
        org = calc_gamma(org)
    # org = contrast_brightness_image(org,0.8,50)
    src = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    src_shape = src.shape
    scale = 2 if src_shape[1]<600 else 3
    ker = (5,5) if src_shape[1]<600 else (7,7)
    # scale = src_shape[1]/300.

    src = cv2.resize(src, (int(src_shape[1]/scale), int(src_shape[0]/scale)))
    org = cv2.resize(org, (int(src_shape[1]/scale), int(src_shape[0]/scale)))
    # src = calc_lap(src)
    # src = cv2.blur(src,(3,3))
    # kernel = np.ones((5, 5), np.float32) / 25
    # src = cv2.filter2D(src, -1, kernel) #灰度图，用于后面的圆圈识别
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    src = cv2.filter2D(src, -1, kernel)
    # src = cv2.equalizeHist(src)


    # src = cv2.bilateralFilter(src, 7, 70, 70)
    # src = cv2.blur(src, (5, 5))
    src = cv2.GaussianBlur(src,ker,1)
    circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 60, param1=60, param2=60, minRadius=30,maxRadius=0)


    for i in circles2[0, :]:
        # draw the outer circle
        cv2.circle(org, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(org, (i[0], i[1]), 2, (0, 0, 255), 3)



    small_circle = sorted(circles2[0][:5], key=lambda x: x[2])[0]

    cv2.circle(org, (small_circle[0], small_circle[1]), small_circle[2], (0, 0, 255), 3)
    cv2.imwrite('./pointer3/2_fit_circle2.jpg', org)
    scale = np.array([scale])

    circle_ = small_circle*scale
    return np.uint16(np.around(circle_))


def findHearts2_test(src,org):
    # org = calc_gamma(org)
    # # org = contrast_brightness_image(org,0.8,50)
    # src = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    src_shape = src.shape
    scale = src_shape[1]/300.
    # src = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    src = cv2.resize(src, (300, 300))
    org = cv2.resize(org, (300, 300))
    # src = calc_lap(src)
    # src = cv2.blur(src,(3,3))
    # kernel = np.ones((5, 5), np.float32) / 25
    # src = cv2.filter2D(src, -1, kernel) #灰度图，用于后面的圆圈识别
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # src = cv2.filter2D(src, -1, kernel)
    # src = cv2.equalizeHist(src)


    src = cv2.bilateralFilter(src, 7, 50, 50)
    # src = cv2.blur(src, (5, 5))
    # src = cv2.GaussianBlur(src,(7,7),1)
    circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=30,maxRadius=0)


    for i in circles2[0, :]:
        # draw the outer circle
        cv2.circle(org, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(org, (i[0], i[1]), 2, (0, 0, 255), 3)



    small_circle = sorted(circles2[0][:5], key=lambda x: x[2])[0]

    cv2.circle(org, (small_circle[0], small_circle[1]), small_circle[2], (0, 0, 255), 3)
    cv2.imwrite('./pointer3/2_fit_circle2.jpg', org)
    scale = np.array([scale])

    circle_ = small_circle*scale
    return np.uint16(np.around(circle_))







def rotateImg(src,avg_angle,heart=None):
    height,width= src.shape
    if heart is None:
        M = cv2.getRotationMatrix2D((int(width/2),int(height/2)), avg_angle, 1.0)
    else:
        M = cv2.getRotationMatrix2D((heart[0],heart[1]), avg_angle, 1.0)
    # heightNew = int(width * math.fabs(math.sin(math.radians(avg_angle))) + height * math.fabs(math.cos(math.radians(avg_angle))))
    # widthNew = int(height * math.fabs(math.sin(math.radians(avg_angle))) + width * math.fabs(math.cos(math.radians(avg_angle))))
    # M[0, 2] += (widthNew - width) / 2
    # M[1, 2] += (heightNew - height) / 2
    num_box = cv2.warpAffine(src, M, (width, height))
    # cv2.imwrite('./pointer2/5_num_box_rotate' + str(ind) + '.jpg', num_box)
    return num_box

def cut_area(src,heart_,type_):

    radio = heart_[2]
    if type_==0:
        hei = radio/3.5
        k = kernel6
        scale = 4
    else:
        hei = radio /2
        k = kernel5
        scale = 4

    mask = np.zeros(src.shape,np.uint8)
    cv2.circle(mask,(heart_[0],heart_[1]),int(radio+hei),255,-1)
    cv2.circle(mask,(heart_[0],heart_[1]),radio-30,0,-1)

    res_img = cv2.bitwise_and(mask,src)
    # res_img = cv2.dilate(res_img0, kernel3)
    cv2.imwrite('./pointer3/3_mask.jpg', res_img)
    if type_==0:
        mask = np.zeros(src.shape, np.uint8)
        cv2.circle(mask, (heart_[0], heart_[1]), int(radio + hei), 255, -1)
        cut = cv2.bitwise_and(mask,src)
        cv2.imwrite('./pointer3/3_mask0.jpg', cut)
    # _, contours, hierarchy = cv2.findContours(res_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # conts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # mask = np.zeros(src.shape, np.uint8)
    # cv2.drawContours(mask,[conts],0,255,-1)
    # cv2.imwrite('./pointer3/3_conts.jpg', mask)
    #
    # res_img = cv2.bitwise_and(mask,res_img0)
    res_img = rotateImg(res_img,50,heart_)


    zoneshape = src.shape
    zone = cv2.resize(res_img, (zoneshape[1] * scale, zoneshape[0] * scale))
    # cv2.imwrite('./v3/6_cut_numZoneCanny.jpg', zone)
    M = zoneshape[1] * 3 / math.log(radio * 3)
    polar = cv2.logPolar(zone, (heart_[0] * scale, heart_[1] * scale), M, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./pointer3/3_polar1.jpg', polar)

    polar_mask = cv2.dilate(polar,kernel7)


    _, contours, hierarchy = cv2.findContours(polar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(c)

    area = polar[y:y+h,x:x+w]
    # area = cv2.dilate(area, kernel4)
    cv2.imwrite('./pointer3/3_area.jpg', area)
    if type_==0:
        return area,cut
    else:
        return area


# def findHearts(src,org):
#     '''
#
#     :param src: 灰度图，用于houghcircle
#     :param org: 原图，彩色图
#     :return: hearts 按半径排序的圆
#     '''
#
#     """
#     找刻度轴的圆心
#     image为输入图像，需要灰度图
#     method为检测方法,常用CV_HOUGH_GRADIENT
#     dp为检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，如dp=1，累加器和输入图像具有相同的分辨率，如果dp=2，累计器便有输入图像一半那么大的宽度和高度
#     minDist表示两个圆之间圆心的最小距离
#     param1有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半
#     param2有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值，它越小，就越可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了
#     minRadius有默认值0，圆半径的最小值
#     maxRadius有默认值0，圆半径的最大值
#     """
#
#     # src = cv2.erode(src,kernel3)
#     # src = cv2.equalizeHist(src)
#     src = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
#     src_shape = src.shape
#     src = cv2.resize(src, (int(src_shape[1] / 2), int(src_shape[0] / 2)))
#     org1 = cv2.resize(org, (int(src_shape[1] / 2), int(src_shape[0] / 2)))
#     src =  cv2.bilateralFilter(src, 5, 50, 50)
#     # adpt = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
#     # ret3, adpt = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#
#     cv2.imwrite('./pointer3/2_cut_find_heart.jpg',src)
#     print('img','./pointer3/2_cut_find_heart.jpg')
#     circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 10, param1=80, param2=60, minRadius=50,maxRadius=0)
#     try:
#
#         # print(len(circles2[0]))
#         if len(circles2[0])<3 or circles2 is None:
#             raise Exception("识别到的圆 少于 3")
#
#         circles2 = np.uint16(np.around(circles2))
#     except Exception as err:
#         print('err',err)
#         circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 60, param1=60, param2=60, minRadius=50, maxRadius=0)
#         circles2 = np.uint16(np.around(circles2))
#
#     if len(circles2[0])<3 or circles2 is None:
#         print('err','未识别到准确的圆')
#         sys.exit(1)
#
#     cuto_cp = org1.copy()
#     for i in circles2[0,:10]:
#         # draw the outer circle
#         cv2.circle(cuto_cp,(i[0],i[1]),i[2],(0,255,0),2)
#         # draw the center of the circle
#         cv2.circle(cuto_cp,(i[0],i[1]),2,(0,0,255),3)
#
#
#     for i in range(0,3):
#         onecircle = circles2[0][i]
#         cv2.circle(cuto_cp,(onecircle[0],onecircle[1]),onecircle[2],(0,0,255),2)
#         # draw the center of the circle
#         cv2.circle(cuto_cp,(onecircle[0],onecircle[1]),2,(0,0,255),3)
#
#     cv2.imwrite('./pointer3/2_cut_Circles.jpg',cuto_cp)
#     print('img', './pointer3/2_cut_Circles.jpg')
#     '''
#     找到圆心位置之后，可以进行极坐标转换：
#     参考 https://zhuanlan.zhihu.com/p/30827442
#     '''
#     # polar = cv2.logPolar(eroded, (circles2[0][1][0], circles2[0][1][1]), 150, cv2.WARP_FILL_OUTLIERS)
#     # polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     # cv2.imwrite('./findpointer/cut_polar.jpg', polar)
#
#     hearts = circles2[0][:3]
#     if hearts[1][2] < hearts[2][2]:
#         tmp = hearts[1].copy()
#         hearts[1] = hearts[2]
#         hearts[2] = tmp
#
#     return hearts

def skeleton(img_adp):
    size = np.size(img_adp)
    ps = cv2.countNonZero(img_adp)
    skel = np.zeros(img_adp.shape, np.uint8)

    # ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    done = False

    while (not done):
        eroded = cv2.erode(img_adp, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_adp, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_adp = eroded.copy()

        rate = float(ps)/cv2.countNonZero(img_adp)
        if rate >= 1:
            done = True

    cv2.imwrite('./pointer3/3_skeleton.jpg',img_adp)
    return img_adp

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


def split_area(src):
    # index=0
    # _areas=[]
    # for src in src_arr:
    #     src1 = src.copy()
    #     src1 = cv2.dilate(src1,kernel8,iterations=2)
    #     un,area,un,un = findContours(src1,src)
    #     area = cv2.erode(area,kernel3)
    #     area = cv2.dilate(area,kernel4)
    #     ret, area = cv2.threshold(area, 127, 255, 0)
    #     lines,un = getLineBorder(area,20,20)
    #     print(lines)
    lines, un = getLineBorder(src, 10, 300)
    print(un)
    src_cp = src.copy()
    for i in lines:
        src_cp[i,:]=255

    cv2.imwrite('./pointer3/4_area' + str(0) + '.jpg', src_cp)
    # test_approx(src)
    nums = src[:lines[2],:]
    kedu = src[lines[1]:,:]
    cv2.imwrite('./pointer3/4_nums' + str(0) + '.jpg', nums)
    kedu = cv2.morphologyEx(kedu,cv2.MORPH_CLOSE,cross1,iterations=4)
    ret, kedu = cv2.threshold(kedu, 100, 255, 0)
    _, contours, hierarchy = cv2.findContours(kedu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(c)

    kedu = kedu[:, x:x + w]
    # lines1, un = getLineBorder(kedu, 10, 50)
    # for i in lines1:
    #     kedu[i,:]=255
    ver = projectVertical(kedu)
    maxtab, mintab = peakdetective.peakdet(ver, 5)
    maxs = maxtab[:, 1]
    tabs = maxtab[:, 0]

    max_pos = np.where(maxs == max(maxs))[0][0]
    print(tabs[max_pos])
    cv2.imwrite('./pointer3/4_kedu' + str(0) + '.jpg', kedu)
        # _areas.append({'nums': black_img(area[lines[0]:lines[1], :]),
        #                'kedu1':black_img(area[lines[2]:lines[3], :]) ,
        #                'kedu2':black_img(area[lines[2]:lines[4], :])})
        # index+=1

    # return

def calc_equalize(src):
    new_img = []
    sp = cv2.split(src)
    for i in range(3):
        new_img.append(cv2.equalizeHist(sp[i]))
    new_img = np.array(new_img)
    new_img = cv2.merge(new_img)
    cv2.imwrite('./pointer3/0_make_equalize.jpg', new_img)
    return new_img

def calc_gamma(src):
    img0 = src
    hist_b = cv2.calcHist([img0], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img0], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img0], [2], None, [256], [0, 256])

    def gamma_trans(img, gamma):
        # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        # 实现映射用的是Opencv的查表函数
        return cv2.LUT(img0, gamma_table)

    img0_corrted = gamma_trans(img0, 0.4)

    cv2.imwrite('./pointer3/0_calc_gamma.jpg', img0_corrted)
    return img0_corrted

def calc_lap(src):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image1 = cv2.filter2D(src, -1, kernel)
    cv2.imwrite('./pointer3/0_lap.jpg', image1)
    return image1

def test_adapt(pth):
    img = cv2.imread(pth)

    # equ = calc_equalize(img)
    equ = calc_gamma(img)
    # equ = calc_lap(img)

    img = cv2.cvtColor(equ, cv2.COLOR_BGR2GRAY)

    #
    # cv2.medianBlur(img, 11)

    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img,(11,11),1)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    # img = cv2.equalizeHist(img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    # img = skeleton(img)
    cv2.imwrite('./pointer3/0_adapt.jpg',img)


def testfillflood(pth):
    src = cv2.imread(pth)
    # blured = src
    # blured = cv2.bilateralFilter(src, 5, 50, 50)


    blured = cv2.blur(src,(3,3))
    canny = cv2.Canny(blured, 80, 100)
    cv2.imwrite('./pointer3/0_canny.jpg', canny)
    h, w = src.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，泛洪填充不会超出掩码的非零边缘
    # 进行泛洪填充
    cv2.floodFill(blured, mask, (int(w) - 1, 1), (255, 255, 255), (2, 2, 2), (10, 10, 10), 8)
    cv2.imwrite('./pointer3/0_fillflood.jpg', blured)
    blured = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    # blured = cv2.erode(blured,cross1,iterations=4)
    blured = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    cv2.imwrite('./pointer3/0_adapt.jpg', blured)




def cut_area2(src,heart_,type_):
    # src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    radio = heart_[2]
    hei = radio / 2
    k = kernel5
    scale = 4
    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart_[0], heart_[1]), int(radio + hei), 255, -1)
    cv2.circle(mask, (heart_[0], heart_[1]), radio - 30, 0, -1)

    res_img = cv2.bitwise_and(mask, src)
    # res_img = cv2.dilate(res_img0, kernel3)
    cv2.imwrite('./pointer3/3_0_mask.jpg', res_img)
    res_img = rotateImg(res_img, 50, heart_)

    zoneshape = src.shape
    zone = cv2.resize(res_img, (zoneshape[1] * scale, zoneshape[0] * scale))
    # cv2.imwrite('./v3/6_cut_numZoneCanny.jpg', zone)
    M = zoneshape[1] * 3 / math.log(radio * 3)
    polar = cv2.logPolar(zone, (heart_[0] * scale, heart_[1] * scale), M, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar = cv2.Canny(polar,80,100)
    cv2.imwrite('./pointer3/3_0_polar1.jpg', polar)


def findPointer(src,heart1,heart2):
    r1 = heart1[2]
    hei1 = r1/3.5
    r2 = heart2[2]
    hei2 = r2 / 2

    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart1[0], heart1[1]), int(r1 + hei1), 255, -1)
    cv2.circle(mask, (heart2[0], heart2[1]), int(r2 + hei2), 0, -1)
    cut1 = cv2.bitwise_and(mask, src)
    cv2.imwrite('./pointer3/3_adp1.jpg', cut1)

    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart1[0], heart1[1]), int(r1 + hei1), 255, -1)
    cut_tmp = cv2.bitwise_and(mask, src)
    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart2[0], heart2[1]), int(r2 + hei2), 255, -1)
    cut2 = cv2.bitwise_and(mask,cut_tmp)
    cv2.imwrite('./pointer3/3_adp2.jpg', cut2)

    h1=r1/2.5
    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart1[0], heart1[1]), int(r1 + hei1), 255, -1)
    cv2.circle(mask, (heart1[0], heart1[1]), int(r1 + hei1/3), 0, -1)
    cut_num1 =cv2.bitwise_and(mask, cut1)
    cv2.imwrite('./pointer3/3_num1.jpg', cut_num1)

    h2 = r2 / 2.5
    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart2[0], heart2[1]), int(r2 + hei2), 255, -1)
    cv2.circle(mask, (heart2[0], heart2[1]), int(r2 + hei2/2.5), 0, -1)
    cut_num2 = cv2.bitwise_and(mask, cut2)
    cv2.imwrite('./pointer3/3_num2.jpg', cut_num2)





    theta1 = detect_pointer(cut1,heart1,1)
    theta2 = detect_pointer(cut2,heart2,2)

    cut1 = make_kedu_area(cut1, heart1,theta1)
    cut2 = make_kedu_area(cut2, heart2,theta2)

    # mask = np.zeros(src.shape, np.uint8)
    # # cv2.ellipse(mask,(heart1[0],heart1[1]),0,0,0,180,255)
    # angle1 = -180/np.pi * theta1
    # angle2 = 180/np.pi * theta2
    # cv2.ellipse(mask, (heart1[0],heart1[1]), (int(r1 + hei1), int(r1 + hei1)), 0, angle1-30, angle1+30, 255, -1)
    # cut1 = cv2.bitwise_and(mask, cut1)
    # cv2.imwrite('./pointer3/3_ellipse.jpg', cut1)
    kedu1,area_sets1 = polar_area(cut1,heart1)

    ver1 = projectVertical(kedu1)
    # print(ver1)
    kedu2,area_sets2 = polar_area(cut2,heart2)
    ver2 = projectVertical(kedu2)

    maxtab1,mintab1 = peakdetective.peakdet(ver1,10)
    maxtab2,mintab2 = peakdetective.peakdet(ver2,10)

    rate1 = findeadge(maxtab1)
    rate2 = findeadge(maxtab2)
    # drawplot(maxtab1,ver1)
    drawplot(maxtab2,ver2)

    polar_num(cut_num1, heart1,area_sets1,1)
    # polar_num(cut_num2,heart2,area_sets2,2)


    return rate1,rate2

def calc_res(rate,index):
    if index==1:
        return 90*rate-30
    else:
        return 100*rate



def drawplot(maxtab2,ver):
    from matplotlib.pyplot import plot, scatter, show


    plot(ver)

    scatter(np.array(maxtab2)[:, 0], np.array(maxtab2)[:, 1], color='blue')
    scatter(np.array([0]), np.array([18]), color='red')
    # scatter(np.array(mintab2)[:, 0], np.array(mintab2)[:, 1], color='red')
    show()



def findeadge(maxtab):
    maxtab1_y = list(maxtab[:, 1])
    maxtab1_x = maxtab[:, 0]
    print(maxtab1_x)
    avg1 = (np.sum(maxtab1_y) - max(maxtab1_y)) / (len(maxtab1_y) - 1)
    left=0
    right=len(maxtab1_x)-1
    print(avg1)
    for i in range(len(maxtab1_x)):
        if maxtab1_y[i]>avg1-5:
            left=i
            break
    for i in range(len(maxtab1_x)-1,0,-1):
        if maxtab1_y[i]>avg1-5:
            right=i
            break

    p = maxtab1_y.index(max(maxtab1_y))

    rate = (maxtab1_x[p]-maxtab1_x[left])/(maxtab1_x[right]-maxtab1_x[left])
    print( maxtab1_x[p],maxtab1_x[left],maxtab1_x[right])
    print(rate)
    return rate





def make_kedu_area(src,heart_,theta_):
    r = heart_[2]
    hei = r / 9
    mask = np.zeros(src.shape, np.uint8)
    cv2.circle(mask, (heart_[0], heart_[1]), int(r + hei), 255, -1)
    cv2.circle(mask, (heart_[0], heart_[1]), int(r-10), 0, -1)
    cut_ = cv2.bitwise_and(mask, src)

    y1 = int(heart_[1] - math.sin(theta_) * heart_[2])
    x1 = int(heart_[0] + math.cos(theta_) * heart_[2])
    cv2.line(cut_, (int((heart_[0]+x1)/2), int((heart_[1]+y1)/2)), (x1, y1), 255, 1)
    cv2.imwrite('./pointer3/3_adp_kedu.jpg', cut_)
    return cut_


def polar_area(src,heart_):
    scale = 4
    res_img = rotateImg(src, 50, heart_)

    zoneshape = src.shape
    zone = cv2.resize(res_img, (zoneshape[1] * scale, zoneshape[0] * scale))
    # cv2.imwrite('./v3/6_cut_numZoneCanny.jpg', zone)
    M = zoneshape[1] * 3 / math.log(heart_[2] * 3)
    polar = cv2.logPolar(zone, (heart_[0] * scale, heart_[1] * scale), M, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./pointer3/3_polar1.jpg', polar)

    polar_mask = cv2.dilate(polar, kernel7)

    _, contours, hierarchy = cv2.findContours(polar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(c)

    area = polar[y:y + h, x:x + w]
    # area = cv2.dilate(area, kernel4)
    cv2.imwrite('./pointer3/3_area.jpg', area)
    return area,[x, y, w, h]


def polar_num(src,heart_,sets,index):
    scale = 4
    res_img = rotateImg(src, 50, heart_)

    zoneshape = src.shape
    zone = cv2.resize(res_img, (zoneshape[1] * scale, zoneshape[0] * scale))
    # cv2.imwrite('./v3/6_cut_numZoneCanny.jpg', zone)
    M = zoneshape[1] * 3 / math.log(heart_[2] * 3)
    polar = cv2.logPolar(zone, (heart_[0] * scale, heart_[1] * scale), M, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    x, y, w, h = sets
    polar = polar[y-int(h/3):y+int(h/3),x:x+w+100]
    polar_mask = cv2.dilate(polar, kernel9)
    cv2.imwrite('./pointer3/4_polar_num.jpg', polar_mask)
    _, contours, hierarchy = cv2.findContours(polar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nums_set=[]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        nums_set.append([x,y,w,h])
    # print(nums_set)
    nums_set = sorted(nums_set,key=lambda a:a[0])
    # print(nums_set)
    x, y, w, h = nums_set[0]
    num1 = polar[y:y + h, x:x + w]

    x, y, w, h = nums_set[-1]
    num2 = polar[y:y + h, x:x + w]

    if index==1:
        num1 = rotateImg(num1,180)
        num2 = rotateImg(num2,180)

    cv2.imwrite('./pointer3/4_num1.jpg', num1)
    cv2.imwrite('./pointer3/4_num2.jpg', num2)
    recog_num(num1)
    # recog_num(num2)
    return nums_set,num1,num2

def getEachNum(src,delta=0):
    (h1, w1) = src.shape
    vertical = projectVertical(src)
    line_border = []

    border_arr = get_border(vertical, delta,5)


    return border_arr, vertical

def recog_num(src):
    src = cv2.dilate(src, kernel3)
    num_border = getEachNum(src)[0]
    tmpArr = []

    maxWidth = 0
    for i in range(0, len(num_border), 2):
        _width = num_border[i + 1] - num_border[i]
        if _width > maxWidth:
            maxWidth = _width

    for i in range(0, len(num_border), 2):
        new_cut = src[:, num_border[i]:num_border[i + 1]]
        blank = np.zeros((new_cut.shape[0], maxWidth), np.uint8)
        # print(new_cut.shape,blank.shape)
        blank[:, blank.shape[1] - new_cut.shape[1]:blank.shape[1]] = new_cut
        new_cut = cv2.resize(blank, (32, 64))
        # cv2.imwrite(dir_path + '/num' + str(index) + '/' + str(_index) + '_' + str(int(i / 2)) + '.jpg', new_cut)
        cv2.imwrite('./pointer3/4_num2_'+str(i)+'.jpg', new_cut)
        # new_cut = Image.fromarray(cv2.cvtColor(new_cut, cv2.cv2.COLOR_BGR2GRAY))
        # hight, width = new_cut.shape
        # new_cut = np.asarray(new_cut)
        # new_cut = new_cut.reshape(1, hight * width)[0]
        x = new_cut.astype(float)
        x *= (1. / 255)
        x = x.reshape(1, 64, 32, 1)
        pr = cnn.predict(x)
        pr = convert2Num(pr)
        print('cnn识别结果：', pr)


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






def detect_pointer(cut,heart_,index):
    mask_max = 0
    mask_theta = 0
    for i in range(0,628):
        black_img = np.zeros(cut.shape, np.uint8)
        theta = float(i) * 0.01
        y1 = int(heart_[1] - math.sin(theta) * heart_[2])
        x1 = int(heart_[0] + math.cos(theta) * heart_[2])
        cv2.line(black_img, (heart_[0], heart_[1]), (x1, y1), 255, 2)
        tmp_img = cv2.bitwise_and(black_img, cut)
        tmp = cv2.countNonZero(tmp_img)
        if tmp > mask_max:
            mask_max = tmp
            mask_theta = theta
    # black_img = np.zeros(src.shape, np.uint8)
    cut_cp = cut.copy()
    y1 = int(heart_[1] - math.sin(mask_theta) * heart_[2])
    x1 = int(heart_[0] + math.cos(mask_theta) * heart_[2])
    cv2.line(cut_cp, (heart_[0], heart_[1]), (x1, y1), 255, 8)
    cv2.imwrite('./pointer3/3_pointer'+str(index)+'.jpg', cut_cp)
    return mask_theta




def load_cnn():
    # model_path = './cnn/num_cnn.h5'
    model_path = './cnn/myCNN_pointer2.h5'
    K.clear_session()  # Clear previous models from memory.
    try:
        cnn_model = load_model(model_path)
    except:
        print('加载模型出错')
        # sys.exit(MODEL_ERR)
    return cnn_model


def test_gray(pth):
    src = cv2.imread(pth)
    src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./pointer3/0_gray0.jpg', src)
    hist_full = cv2.calcHist([src], [0], None, [256], [0, 256])
    plt.plot(hist_full)
    plt.show()

    rows, cols = src.shape
    flat_gray = src.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' % (A, B))
    gray = np.uint8(255 / (B - A) * (src - A) + 0.5)
    cv2.imwrite('./pointer3/0_gray.jpg', gray)
    # hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # plt.plot(hist_full)
    # plt.show()


# if __name__ == '__main__':
#     # path = './位置2/30/image3.jpg'
#     # path = './位置4/image3.jpg' 1/6 2/15 2/17
#     # for i in range(40):2/14 2/16 2/17 2/19
#         path = './pointer3_img/1/img'+str(4)+'.jpg'
#         # test_adapt(path)
#         test_gray(path)

if __name__ == '__main__':
    # path = './位置2/30/image3.jpg'
    # path = './位置4/image3.jpg' 1/6 2/15 2/17
    # for i in range(40):2/14 2/16 2/17 2/19
        path = './pointer3_img/1/img'+str(4)+'.jpg'
        # test_adapt(path)
        # test_gray(path)
        # testfillflood(path)
        # dark = False
        cut_Img, cut_origin, center1 = findMainZone(path)
        heart_1 = findHearts1(cut_Img,center1)
        heart_2 = findHearts2(cut_origin)
        # cut_Img=  skeleton(cut_Img)
        # area1,cut_Img = cut_area(cut_Img,heart_1,0)
        print('4.提前加载模型')
        cnn = load_cnn()
        ra1,ra2 = findPointer(cut_Img,heart_1,heart_2)
        res1 = calc_res(ra1,1)
        res2 = calc_res(ra2,2)
        print(res1,res2)
        # area2 =  cut_area(cut_Img,heart_2,1)
        # # # split_area(area1)
        # split_area(area2)



