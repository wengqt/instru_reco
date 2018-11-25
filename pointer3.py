import cv2
import numpy as np
import sys
import math
import peakdetective

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernel6 = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
kernel8 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))


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
    cutImg_o = img1[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    # gray_o = canny[d1:the_circle[1] + the_circle[2],
    #            d2:the_circle[0] + the_circle[2]]
    cv2.imwrite('./pointer3/1_cut.jpg', cutImg_o)
    print('img', './pointer3/1_cut.jpg')

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    # cut_g = cv2.GaussianBlur(cutImg, (7, 7), 1)
    # cutImg = cv2.equalizeHist(cutImg)
    # cutCanny = cv2.Canny(cut_g, 50, 100)
    # cv2.imwrite(dir_path+'/1_cut_Canny.jpg', cutCanny)
    kernel = np.ones((3, 3), np.float32) / 25
    cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    # cutImg1 = cv2.bilateralFilter(cutImg, 9, 100, 100)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),1) #灰度图，用于后面的圆圈识别
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别

    cutImg3 = cv2.adaptiveThreshold(cutImg1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    # cutImg3 = cv2.erode(cutImg3,kernel3,iterations=1)
    # ret2, cutImg3 = cv2.threshold(cutImg1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cutImg3 = cv2.erode(cutImg3,kernel3)
    cv2.imwrite('./pointer3/1_cut_adapt.jpg', cutImg3)
    print('img', './pointer3/1_cut_adapt.jpg')
    return cutImg3,cutImg_o,(the_circle[0]-d2,the_circle[1]-d1)



# def findCircleContours(src,big_index=0,points_num=20):
#     _, contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#


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
    src = cv2.erode(src, kernel3)

    src = cv2.dilate(src,kernel7)

    src = cv2.erode(src, kernel8,iterations=1)
    cv2.imwrite('./pointer3/2_erode.jpg', src)
    # cv2.imwrite('./pointer3/2_erode.jpg', src)

    src_shape = src.shape

    fit_r = 0
    max_mean = 0
    for r in range(int(src_shape[0]/4.),int(src_shape[0]/2.5)):
        black = np.zeros(src_shape,np.uint8)
        cv2.circle(black,heart1,r,255,10)
        tmp = np.mean(cv2.bitwise_and(black, src))
        if tmp>max_mean:
            max_mean = tmp
            fit_r = r

    cv2.circle(src, heart1, fit_r, 255, 10)
    cv2.imwrite('./pointer3/2_fit_circle.jpg', src)



    return [heart1[0],heart1[1],fit_r]


def findHearts2(org):
    src = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    src_shape = src.shape
    src = cv2.resize(src, (int(src_shape[1] / 2), int(src_shape[0] / 2)))
    org = cv2.resize(org, (int(src_shape[1] / 2), int(src_shape[0] / 2)))
    # kernel = np.ones((5, 5), np.float32) / 25
    # src = cv2.filter2D(src, -1, kernel) #灰度图，用于后面的圆圈识别
    src = cv2.bilateralFilter(src, 5, 50, 50)
    circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=40, param2=60, minRadius=30,maxRadius=0)


    for i in circles2[0, :]:
        # draw the outer circle
        cv2.circle(org, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(org, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imwrite('./pointer3/2_fit_circle2.jpg', org)

    small_circle = sorted(circles2[0][:5], key=lambda x: x[2])[0]
    scale = np.array([2])

    circle_ = small_circle*scale
    return np.uint16(np.around(circle_))

def rotateImg(src,avg_angle,heart):
    height,width= src.shape
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
        hei = radio /3
        k = kernel5
        scale = 4

    mask = np.zeros(src.shape,np.uint8)
    cv2.circle(mask,(heart_[0],heart_[1]),int(radio+hei),255,-1)
    cv2.circle(mask,(heart_[0],heart_[1]),radio-30,0,-1)

    res_img = cv2.bitwise_and(mask,src)
    # res_img = cv2.dilate(res_img0, kernel3)
    cv2.imwrite('./pointer3/3_mask.jpg', res_img)

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
    lines, un = getLineBorder(src, 10, 50)
    src_cp = src.copy()
    for i in lines:
        src_cp[i,:]=255

    cv2.imwrite('./pointer3/4_area' + str(0) + '.jpg', src_cp)

    nums = src[:lines[2],:]
    kedu = src[lines[1]:,:]
    cv2.imwrite('./pointer3/4_nums' + str(0) + '.jpg', nums)
    kedu = cv2.dilate(kedu,kernel4,iterations=2)
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


if __name__ == '__main__':
    # path = './位置2/30/image3.jpg'
    # path = './位置4/image3.jpg'
    # for i in range(40):
        path = './pointer3_img/img'+str(8)+'.jpg'
        cut_Img, cut_origin, center1 = findMainZone(path)
        heart_1 = findHearts1(cut_Img,center1)
        heart_2 = findHearts2(cut_origin)
        cut_Img=  skeleton(cut_Img)
        # area1 = cut_area(cut_Img,heart_1,0)
        area2 =  cut_area(cut_Img,heart_2,1)
        # split_area(area1)
        split_area(area2)


