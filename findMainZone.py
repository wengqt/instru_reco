import cv2
import numpy as np
import sys





def findMainZone(path):
    # print(path)
    img1 = cv2.imread(path)
    try:
        if img1 is None:
            raise Exception('找不到图片! 图片路径有误。')
    except Exception as err:
        print(err)
        sys.exit(1)
    img1shape = img1.shape
    img1 = cv2.resize(img1, (int(img1shape[1]/2), int(img1shape[0]/2)))


    # canny = cv2.Canny(cv2.GaussianBlur(img1, (7, 7), 0), 100, 250)
    # cv2.imwrite('./v3/canny.jpg', canny)

    canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    '''
    找到仪表区域
    '''
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=50)
    circles = np.uint16(np.around(circles))
    img1_cp = img1.copy()
    for i in circles[0, 0:3]:
        # 画圆圈
        cv2.circle(img1_cp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img1_cp, (i[0], i[1]), 2, (0, 0, 255), 3)

    _circles = circles[0,:3] #投票数最大的圆
    _circles = sorted(_circles,key=lambda x:x[2],reverse=True )
    # print(_circles)
    the_circle = _circles[0]
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), the_circle[2], (0, 0, 255), 2)
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), 2, (0, 0, 255), 3)
    cv2.imwrite('./v3/1_circles.jpg', img1_cp)
    # print(the_circle[1],the_circle[2])
    # print(int(the_circle[1]) - int(the_circle[2]))
    d1 =0 if int(the_circle[1]) - int(the_circle[2])<0 else int(the_circle[1]) - int(the_circle[2])
    d2 =0 if int(the_circle[0]) - int(the_circle[2])<0 else int(the_circle[0]) - int(the_circle[2])
    cutImg_o = img1[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    gray_o = canny[d1:the_circle[1] + the_circle[2],
               d2:the_circle[0] + the_circle[2]]
    cv2.imwrite('./v3/1_cut.jpg', cutImg_o)

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    # cut_g = cv2.GaussianBlur(cutImg, (7, 7), 1)
    # cut_g = cv2.equalizeHist(cut_g)
    # cutCanny = cv2.Canny(cut_g, 50, 100)
    # cv2.imwrite('./v3/1_cut_Canny.jpg', cutCanny)
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg1 = cv2.bilateralFilter(cutImg, 9, 70, 70)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),0) #灰度图，用于后面的圆圈识别
    kernel = np.ones((3, 3), np.float32) / 25
    cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别

    cutImg3 = cv2.adaptiveThreshold(cutImg2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    # ret2, cutImg3 = cv2.threshold(cutImg2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('./v3/1_cut_adapt.jpg', cutImg3)
    return cutImg3,cutImg_o,cutImg1,gray_o

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
    cv2.imwrite('./v3/2_cut_find_heart.jpg',src)
    circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 100, param1=80, param2=60, minRadius=50,maxRadius=0)
    try:

        # print(len(circles2[0]))
        if len(circles2[0])<3 or circles2 is None:
            raise Exception("识别到的圆 少于 3")

        circles2 = np.uint16(np.around(circles2))
    except Exception as err:
        print(err)
        circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=60, param2=60, minRadius=50, maxRadius=0)
        circles2 = np.uint16(np.around(circles2))

    if len(circles2[0])<3 or circles2 is None:
        print('未识别到准确的圆')
        # sys.exit(CIRCLE_ERR)

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

    cv2.imwrite('./v3/2_cut_Circles.jpg',cuto_cp)
    '''
    找到圆心位置之后，可以进行极坐标转换：
    参考 https://zhuanlan.zhihu.com/p/30827442
    '''
    # polar = cv2.logPolar(eroded, (circles2[0][1][0], circles2[0][1][1]), 150, cv2.WARP_FILL_OUTLIERS)
    # polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite('./findpointer/cut_polar.jpg', polar)

    hearts = circles2[0][:3]
    if hearts[1][2] < hearts[2][2]:
        tmp = hearts[1].copy()
        hearts[1] = hearts[2]
        hearts[2] = tmp

    return hearts

if __name__ == '__main__':
    # path = './位置2/30/image3.jpg'
    # path = './位置4/image3.jpg'
    path = './img/image2.jpg'
    cut_Img, cut_origin, grayImg,gray_origin = findMainZone(path)

    print('2.找圆心')
    heartsArr = findHearts(gray_origin, cut_origin)