
import cv2
import numpy as np


def findMainZone(path):
    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (3000, 2000))
    canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0),cv2.COLOR_BGR2GRAY)
    # canny = cv2.Canny(cv2.GaussianBlur(img1, (7, 7), 0), 100, 250)
    # cv2.imwrite('./findhearts/canny.jpg', canny)
    '''
    找到仪表区域
    '''
    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=50)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=50)
    circles = np.uint16(np.around(circles))
    img1_cp = img1.copy()
    for i in circles[0, :]:
        # 画圆圈
        cv2.circle(img1_cp, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img1_cp, (i[0], i[1]), 2, (0, 0, 255), 3)

    the_circle = circles[0][0] #投票数最大的圆
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), the_circle[2], (0, 0, 255), 2)
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), 2, (0, 0, 255), 3)
    cv2.imwrite('./findhearts/circles.jpg', img1_cp)

    cutImg_o = img1[the_circle[1] - the_circle[2]:the_circle[1] + the_circle[2],
               the_circle[0] - the_circle[2]:the_circle[0] + the_circle[2]]
    cv2.imwrite('./findhearts/cut.jpg', cutImg_o)

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    cutCanny = cv2.Canny(cv2.GaussianBlur(cutImg, (5, 5), 0), 50, 100)
    cv2.imwrite('./findhearts/cut_Canny.jpg', cutCanny)
    kernel = np.ones((5, 5), np.float32) / 25
    cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cv2.imwrite('./findhearts/cut_filter1.jpg', cutImg1)
    cutImg2 = cv2.GaussianBlur(cutImg, (7,7), 1) #灰度图，用于后面的圆圈识别
    cv2.imwrite('./findhearts/cut_filter2.jpg', cutImg2)
    cutImg3 = cv2.bilateralFilter(cutImg, 9, 70,70) #灰度图，用于后面的圆圈识别
    cv2.imwrite('./findhearts/cut_filter3.jpg', cutImg3)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),0) #灰度图，用于后面的圆圈识别
    cutImg = cv2.adaptiveThreshold(cutImg1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./findhearts/cut_adapt1.jpg', cutImg)
    cutImg = cv2.adaptiveThreshold(cutImg2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./findhearts/cut_adapt2.jpg', cutImg)
    cutImg = cv2.adaptiveThreshold(cutImg3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./findhearts/cut_adapt3.jpg', cutImg)
    return cutImg,cutCanny,cutImg_o,cutImg1,cutImg2,cutImg3


def findHearts(src,org,index):
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

    '''
    findHearts
    src 为hough检测的图片
    org 为
    '''
    # src = cv2.erode(src,kernel3)
    # src = cv2.equalizeHist(src)
    # cv2.imwrite('./findhearts/cut_equalize'+str(index)+'.jpg',src)
    circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=60, minRadius=50,maxRadius=0)
    try:
        circles2 = np.uint16(np.around(circles2))
        if len(circles2)<3:
            raise Exception("circle 少于 3")
    except:
        circles2 = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 50, param1=60, param2=60, minRadius=50, maxRadius=0)

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

    cv2.imwrite('./findhearts/cut_Circles'+str(index)+'.jpg',cuto_cp)
    '''
    找到圆心位置之后，可以进行极坐标转换：
    参考 https://zhuanlan.zhihu.com/p/30827442
    '''
    # polar = cv2.logPolar(eroded, (circles2[0][1][0], circles2[0][1][1]), 150, cv2.WARP_FILL_OUTLIERS)
    # polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite('./v1/cut_polar.jpg', polar)

    return [circles2[0][1],circles2[0][2]]




if __name__ == '__main__':

    #查找仪表圆形区域
    print('查找仪表圆形区域')
    cut_Img,cut_canny,cut_origin,grayImg1,grayImg2,grayImg3 = findMainZone('./位置1/35/image1.jpg')
    # cut_Img,cut_canny,cut_origin,grayImg1,grayImg2,grayImg3 = findMainZone('./image1.jpg')
    #指针角度计算
    # calcAngle(ang1)
    # 找圆心
    print('找圆心')
    heartsArr = findHearts(grayImg1, cut_origin,1)
    heartsArr = findHearts(grayImg2, cut_origin,2)
    heartsArr = findHearts(grayImg3, cut_origin,3)
