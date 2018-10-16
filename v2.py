import cv2
import numpy as np
import math
import peakdetective
import softmax
from keras.models import load_model

from keras import backend as K
from PIL import Image

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernel6 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))


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
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(c[big_index])
    new_img =None
    if dst is not None:
        cv2.rectangle(dst, (x-offset1, y-offset1), (x + w+offset2, y + h+offset2), (0, 255, 0), 2)
        new_img = dst[y-offset1:y + h+offset2,x-offset1:x + w+offset2]
    new_img_canny = dstCanny[y-offset1:y + h+offset2,x-offset1:x + w+offset2]
    cv2.imwrite('./v1/cut_findContours.jpg', new_img_canny)
    if heartsarr is not None:
        for a in heartsarr:
            a[0] =a[0]-x+offset1
            a[1] = a[1] - y + offset1
    return new_img,new_img_canny,heartsarr





# cutImg = cv2.GaussianBlur(cutImg, (5, 5), 0)

# cutImg = cv2.equalizeHist(cutImg)
# cv2.imwrite('./v1/cut_enhence.jpg',cutImg)









# print(avgAngles)


# lines = cv2.HoughLines(eroded,1,np.pi/180,100)
# lines1 = lines[:,0,:]
#
# for rho,theta in lines1[:]:
#     if abs( theta) < np.pi / 6 :
#         print('a angle is :',theta*180/np.pi,'and ',(np.pi / 2 - theta)*180/np.pi)
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x11 = int(x0 + 1000*(-b))
#         y11 = int(y0 + 1000*(a))
#         x22 = int(x0 - 1000*(-b))
#         y22 = int(y0 - 1000*(a))
#         cv2.line(cutImg_1,(x11,y11),(x22,y22),(0,0,255),3)
# cv2.imwrite('./v1/line1.jpg',cutImg_1)


def findHearts(src,org):
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
    cv2.imwrite('./v1/cut_find_heart.jpg',src)
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

    cv2.imwrite('./v1/cut_Circles.jpg',cuto_cp)
    '''
    找到圆心位置之后，可以进行极坐标转换：
    参考 https://zhuanlan.zhihu.com/p/30827442
    '''
    # polar = cv2.logPolar(eroded, (circles2[0][1][0], circles2[0][1][1]), 150, cv2.WARP_FILL_OUTLIERS)
    # polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite('./v1/cut_polar.jpg', polar)

    return [circles2[0][1],circles2[0][2]],circles2[0]










def getLineBorder(src,min_range=0):
    (h1, w1) = src.shape
    horizon = [0 for z in range(0, h1)]

    for i in range(0, h1):  # 遍历一行
        for j in range(0, w1):  # 遍历一列
            if src[i, j] != 0:
                horizon[i] += 1

    line_border = []
    def getLineZone1(src, n1, n2):
        for i in range(n1, n2):
            # print(i)
            if src[i] != 0:
                t = i if i>0 else 0
                line_border.append(t)
                getLineZone2(src, t, n2)
                break
    def getLineZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] <= 20:
                t = i+10 if i+10<=n2 else n2
                line_border.append(t)
                getLineZone1(src, t, n2)
                break
            elif i>=n2-1:
                line_border.append(n2-1)

    getLineZone1(horizon, 0, h1)
    print('lineborder',line_border)

    tmp=[]
    for i in range(0,len(line_border),2):
        if line_border[i+1]-line_border[i]>min_range:
            tmp.append(line_border[i])
            tmp.append(line_border[i+1])

    return tmp,horizon







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
    def getLineZone1(src, n1, n2):
        for i in range(n1, n2):
            # print(i)
            if src[i] != 0:
                t = i if i > 0 else 0
                line_border.append(t)
                getLineZone2(src, i, n2)
                break

    def getLineZone2(src, n1, n2):
        for i in range(n1, n2):
            if src[i] == 0:
                t = i if i <= n2 else n2
                line_border.append(t)
                getLineZone1(src, i, n2)
                break
            elif i >= n2:
                line_border.append(n2)

    getLineZone1(vertical, 0, w1)
    # print('numborder', line_border)
    tmp = []
    for i in range(0, len(line_border), 2):
        if line_border[i + 1] - line_border[i] > delta:
            tmp.append(line_border[i])
            tmp.append(line_border[i + 1])
    return tmp, vertical




def cutNums(nums_Arr,cut,path):
    cut_num = []
    for i in range(0,len(nums_Arr),2):
        new_cut = cut[:,nums_Arr[i]:nums_Arr[i+1]]
        cut_num.append(new_cut)
        # new_cut = cv2.erode(new_cut, kernel5)
        cv2.imwrite(path+str(int(i/2))+'.jpg', new_cut)

    return cut_num








# all=0
# short =0
# for i in ver:
#     if i != 0 and all!= 0:
#         all+= 1
#     elif i!= 0 and all== 0:
#         all+= 1
#         short = ver.index(i)
#
# posi = (ver.index(max(ver))-short)/all
#
# print('posi',posi,all,ver.index(max(ver))+1)






# from matplotlib.pyplot import plot, scatter, show
# plot(ver)
# scatter(np.array(maxtab)[:, 0], np.array(maxtab)[:, 1], color='blue')
# scatter(np.array(mintab)[:, 0], np.array(mintab)[:, 1], color='red')
# show()


'''
处理小的表盘
'''
def getNonPointer(cutImg,pointerImg):
    non_p = cv2.subtract(cutImg, cv2.dilate(pointerImg, kernel3, iterations=2))
    cv2.imwrite('./v1/non_pointer.jpg', non_p)
    return non_p


def processNonPointer(non_p,one_heart):
    # unused, non_numZone_adp, unused = findContours(cut_mask, non_p, offset1=85, offset2=70)
    zoneshape = non_p.shape
    non_numZone_adp = cv2.resize(non_p, (zoneshape[1] * 8, zoneshape[0] * 8))
    cv2.imwrite('./v1/x_cut_non_numZoneCanny.jpg', non_numZone_adp)
    polar_ = cv2.logPolar(non_numZone_adp, (one_heart[0] * 8, one_heart[1] * 8), 600, cv2.WARP_FILL_OUTLIERS)
    polar_ = cv2.rotate(polar_, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./v1/x_cut_polar_1_non.jpg', polar_)
    unused, numsCanny_non, unused1 = findContours(cv2.dilate(polar_, kernel5, iterations=2), polar_, offset1=40,
                                                  offset2=30, big_index=0)
    numsShape = numsCanny_non.shape
    numsCanny_non = cv2.resize(numsCanny_non, (numsShape[1] * 4, numsShape[0] * 4))
    threshold, numsCanny_non = cv2.threshold(numsCanny_non, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    numsCanny_non = cv2.erode(numsCanny_non,kernel3,iterations=1)
    cv2.imwrite('./v1/x_nums_canny1_non.jpg', numsCanny_non)
    line_border_, ho = getLineBorder(numsCanny_non, 20)
    kedu_ = numsCanny_non[line_border_[0]:line_border_[1], :]
    # print(ho)
    return numsShape,polar_,line_border_,kedu_,numsCanny_non,ho


def processSmallKedu(zone2polar,one_heart,numsShape,polar_,line_border_):
    polar = cv2.logPolar(zone2polar, (one_heart[0]*8, one_heart[1]*8), 600, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./v1/x_cut_polar_1.jpg',polar)
    unused,numsCanny_1,unused1=findContours(cv2.dilate(polar_,kernel5,iterations=2),polar,offset1=40,offset2=30,big_index=0)
    numsCanny_1 = cv2.resize(numsCanny_1,(numsShape[1]*4,numsShape[0]*4))
    cv2.imwrite('./v1/x_nums_canny1.jpg',numsCanny_1)
    kedu_1_pointer = numsCanny_1[line_border_[0]:line_border_[1]+60,:]
    kedu_1_pointer = cv2.erode(kedu_1_pointer,kernel3)
    cv2.imwrite('./v1/x_nums_kedu_1_pointer.jpg',kedu_1_pointer)
    ver_ = projectVertical(kedu_1_pointer)
    (h1, w1) = kedu_1_pointer.shape
    newHorizon_ = np.zeros([h1, w1], np.uint8)

    for i in range(0, w1):
        for j in range(0, ver_[i]):
            newHorizon_[j, i] = 255

    cv2.imwrite('./v1/x_nums_kedu_111.jpg',newHorizon_)
    maxtab,mintab = peakdetective.peakdet(ver_,30)
    # from matplotlib.pyplot import plot, scatter, show
    # plot(ver)
    # scatter(np.array(maxtab)[:, 0], np.array(maxtab)[:, 1], color='blue')
    # scatter(np.array(mintab)[:, 0], np.array(mintab)[:, 1], color='red')
    # show()
    k_res = list(maxtab[:,1])
    # print(res.index(max(res)),len(res))
    k_pos =  k_res.index(max(k_res))
    k_len = len(k_res)
    print('position',k_res.index(max(k_res)),'总刻度线数：',(len(k_res)-1))
    return k_pos,k_len

#处理数字
def processSmallNum(numsCanny_non,line_border_,k_pos,k_len):
    numsimg_ = numsCanny_non[line_border_[2]:line_border_[3],:]
    numsimg_ = cv2.erode(numsimg_,kernel5)

    numsdilate_ = cv2.dilate(numsimg_,kernel5,iterations=6)
    cv2.imwrite('./v1/xi_numsimg_.jpg',numsdilate_)
    numsArr_ = getEachNum(numsdilate_,60)[0]


    nums_ = cutNums(numsArr_,numsimg_,'./v1/nums1/')
    index_=0

    numbers_ =[]
    for one_num in nums_:
        num_border = getEachNum(one_num)[0]
        tmpArr = []
        for i in range(0,len(num_border),2):
            # print(num_border[i],num_border[i+1])
            new_cut = one_num[:,num_border[i]:num_border[i+1]]
            new_cut=cv2.resize(new_cut,(64,128))
            cv2.imwrite('./v1/nums1/' +str(index_)+'_'+ str(int(i / 2)) + '.jpg', new_cut)
            # new_cut = Image.fromarray(cv2.cvtColor(new_cut, cv2.cv2.COLOR_BGR2GRAY))
            new_cut = np.asarray(new_cut)
            # new_cut = new_cut.reshape(1, hight * width)[0]

            tmpArr.append(new_cut)
        index_ += 1
        numbers_.append(tmpArr)




    testDigits = [numbers_[0],numbers_[len(numbers_)-1]]

    kedu_range = []
    for test in testDigits:
        res = ''
        for i in test:


            x = cv2.resize(i,(32,64))
            x = x.astype('float32')
            x /= 255
            x = x.reshape(1, 64, 32, 1)
            pr = cnn.predict(x)
            predict =  convert2Num(pr)
            print('cnn识别结果：', convert2Num(pr))
            # i = i.astype('float32')
            # i /= 255
            # i = i.reshape(1, 128 * 64)[0]
            # predict = slearn.predict(i)

            res += '-' if str(predict)=='10' else str(predict)
        print('识别为：',res)
        kedu_range.append(int(res))

    result = k_pos*(k_len-1)/(kedu_range[1]-kedu_range[0])+kedu_range[0]
    print('结果：',result)

def learnNums():
    softmax_learn = softmax.Softmax()
    trainDigits, trainLabels = softmax_learn.loadData('./train')
    softmax_learn.train(trainDigits, trainLabels, maxIter=100)  # 训练
    return softmax_learn


def findMainZone(path):
    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (3000, 2000))
    # canny = cv2.Canny(cv2.GaussianBlur(img1, (7, 7), 0), 100, 250)
    # cv2.imwrite('./v1/canny.jpg', canny)

    canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    '''
    找到仪表区域
    '''
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
    cv2.imwrite('./v1/1_circles.jpg', img1_cp)

    cutImg_o = img1[the_circle[1] - the_circle[2]:the_circle[1] + the_circle[2],
               the_circle[0] - the_circle[2]:the_circle[0] + the_circle[2]]
    cv2.imwrite('./v1/1_cut.jpg', cutImg_o)

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    cutCanny = cv2.Canny(cv2.GaussianBlur(cutImg, (5, 5), 0), 50, 100)
    cv2.imwrite('./v1/1_cut_Canny.jpg', cutCanny)
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg1 = cv2.bilateralFilter(cutImg, 9, 70, 70)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),0) #灰度图，用于后面的圆圈识别
    kernel = np.ones((3, 3), np.float32) / 25
    cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg = cv2.adaptiveThreshold(cutImg2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./v1/1_cut_adapt.jpg', cutImg)
    return cutImg,cutCanny,cutImg_o,cutImg1

# def findPointer(adaptImg):
#     eroded = cv2.erode(adaptImg, kernel3)
#     cv2.imwrite('./v1/2_cut_eroded.jpg', eroded)
#
#     sap = adaptImg.shape
#     black_img = np.zeros([sap[0], sap[1]], np.uint8)
#     lines = cv2.HoughLinesP(eroded, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     lines1 = lines[:, 0, :]  # 提取为二维
#
#     angles1 = []
#     for x1, y1, x2, y2 in lines1[:]:
#         # print(x1,y1,x2,y2)
#         if x2 - x1 != 0:
#             thet = math.atan(abs(y2 - y1) / abs(x2 - x1)) * 180 / np.pi
#             print(thet)
#             if thet >= 30 and thet <= 90:
#                 # print(thet)
#                 angles1.append(thet)
#                 cv2.line(black_img, (x1, y1), (x2, y2), 255, 1)
#
#     cv2.imwrite('./v1/2_lines.jpg', black_img)
#     return black_img,angles1


def findPointer(adaptImg):
    # eroded = cv2.erode(adaptImg, kernel3)
    # eroded = cv2.dilate(eroded, kernel4)
    # eroded = cv2.morphologyEx(adaptImg,cv2.MORPH_OPEN,kernel3)
    # eroded = cv2.Canny(adaptImg,50,100)
    # eroded = cv2.dilate(eroded, kernel5)
    # eroded = cv2.erode(eroded, kernel5)
    eroded = adaptImg
    eroded = cv2.dilate(eroded, kernel3)
    cv2.imwrite('./findpointer/2_cut_eroded.jpg', eroded)

    sap = adaptImg.shape
    black_img = np.zeros([sap[0], sap[1]], np.uint8)
    lines = cv2.HoughLinesP(eroded, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    print(lines)
    lines1 = lines[:, 0, :]  # 提取为二维

    angles1 = []
    for x1, y1, x2, y2 in lines1[:]:
        # print(x1,y1,x2,y2)
        if x2 - x1 != 0:
            thet = math.atan(abs(y2 - y1) / abs(x2 - x1)) * 180 / np.pi
            print(thet)
            if thet >= 30 and thet <= 90:
                # print(thet)
                angles1.append(thet)
                cv2.line(black_img, (x1, y1), (x2, y2), 255, 1)

    cv2.imwrite('./findpointer/2_lines.jpg', black_img)


    return black_img





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

def getBigArea(numZone_adp,hearts):
    # eroded2 = cv2.erode(none_pointer_img, kernel3,iterations=2)
    # eroded2 = cv2.dilate(eroded2, kernel5, iterations=5)
    # cv2.imwrite('./v1/cut_dilate2.jpg', eroded2)
    # unused, numZone_adp, hearts = findContours(eroded2, need_cut, hearts, offset1=85, offset2=70)
    zoneshape = numZone_adp.shape
    numZone_adp = cv2.resize(numZone_adp, (zoneshape[1] * 8, zoneshape[0] * 8))
    cv2.imwrite('./v1/7_cut_numZoneCanny.jpg', numZone_adp)
    # 极坐标转换
    polar = cv2.logPolar(numZone_adp, (hearts[0][0] * 8, hearts[0][1] * 8), 600, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./v1/7_cut_polar.jpg', polar)

    unused, numsCanny, unused1 = findContours(cv2.dilate(polar, kernel5, iterations=1), polar, offset1=10, offset2=20)

    numsShape = numsCanny.shape
    numsCanny = cv2.resize(numsCanny, (numsShape[1] * 4, numsShape[0] * 4))
    threshold, numsCanny = cv2.threshold(numsCanny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('./v1/7_nums_canny.jpg', numsCanny) #直着的 带刻度带数字图
    return numZone_adp,hearts,numsCanny

def convert2Num(onehot):
    '''

    :param onehot: type nparray only
    :return:
    '''

    p= np.where(onehot==np.max(onehot))
    # print(onehot)
    # print('可能性：',onehot[p])
    return str(int(p[1][0]))


def processBigNum(lineZone,numlines,k_pos,k_len):

    numsimg = lineZone[numlines[0]:numlines[1], :]
    numsdilate = cv2.erode(numsimg, kernel4, iterations=2)
    numsdilate = cv2.dilate(numsdilate, kernel5, iterations=3)
    cv2.imwrite('./v1/9_numsimg.jpg', numsdilate)
    numsArr = getEachNum(numsdilate)[0]
    nums = cutNums(numsArr, numsimg, './v1/nums/')
    index = 0
    numbers_ = []
    for one_num in nums:
        num_border = getEachNum(one_num)[0]
        tmpArr = []
        for i in range(0, len(num_border), 2):
            new_cut = one_num[:, num_border[i]:num_border[i + 1]]
            new_cut = cv2.resize(new_cut, (64, 128))
            cv2.imwrite('./v1/nums/' + str(index) + '_' + str(int(i / 2)) + '.jpg', new_cut)
            # new_cut = Image.fromarray(cv2.cvtColor(new_cut, cv2.cv2.COLOR_BGR2GRAY))
            new_cut = np.asarray(new_cut)
            # new_cut = new_cut.reshape(1, hight * width)[0]
            tmpArr.append(new_cut)
        index += 1
        numbers_.append(tmpArr)

    testDigits = [numbers_[0], numbers_[len(numbers_) - 1]]
    kedu_range = []
    for test in testDigits:
        res = ''
        for i in test:
            x = cv2.resize(i,(32,64))
            x = x.astype(float)
            x *= (1. / 255)
            # x = np.array([x])
            x = x.reshape(1, 64, 32, 1)
            pr = cnn.predict(x)
            predict = convert2Num(pr)
            print('cnn识别结果：',convert2Num(pr))
            # i = i.astype(float)
            # i /= 255
            # i = i.reshape(1,128*64)[0]
            # predict = slearn.predict(i)
            res += '-' if str(predict) == '10' else str(predict)
        print('识别为:',res)
        kedu_range.append(int(res))

    result = k_pos * (k_len - 1) / (kedu_range[1] - kedu_range[0]) + kedu_range[0]
    print('结果：',result)




def processBigKedu(lineZone):
    border, ho = getLineBorder(lineZone, 20)
    lineZone = cv2.erode(lineZone, kernel3)
    numsCannyshape = lineZone.shape
    kedu = lineZone[border[2]:border[2]+500, :]
    cv2.imwrite('./v1/8_nums_kedu.jpg', kedu)
    ver = projectVertical(kedu)
    (h1, w1) = kedu.shape
    newHorizon = np.zeros([h1, w1], np.uint8)
    for i in range(0, w1):
        for j in range(0, ver[i]):
            newHorizon[j, i] = 255
    cv2.imwrite('./v1/8_nums_kedu_.jpg', newHorizon)
    maxtab, mintab = peakdetective.peakdet(ver, 3)

    k_res = list(maxtab[:, 1])
    # print(res.index(max(res)),len(res))
    k_pos = k_res.index(max(k_res))
    k_len = len(k_res)
    print('指针位置',k_res.index(max(k_res)), '总刻度线个数',len(k_res))
    # print('position', k_res.index(max(k_res)) / (len(k_res) - 1) * 100)

    return border,k_pos,k_len



def getScaleArea(heart_arr,_img,none_pointer_img):
    h,w = _img.shape
    hearts = heart_arr[:3]
    if hearts[1][2]<hearts[2][2]:
        tmp=hearts[1].copy()
        hearts[1]=hearts[2]
        hearts[2] = tmp
    print(hearts)
    blank_img = np.zeros((h,w),np.uint8)
    r1=hearts[2][1]-hearts[0][1]
    o1=hearts[2]
    r2 = int(hearts[2][2]-r1 + hearts[1][2])
    o2 = hearts[1]

    blank_img = cv2.circle(blank_img,(o2[0],o2[1]),r2,255,-1)
    blank_img = cv2.circle(blank_img, (o1[0], o1[1]), r1, 0, -1)
    _img = cv2.bitwise_and(blank_img, _img)
    none_pointer_img = cv2.bitwise_and(blank_img, none_pointer_img)
    cv2.imwrite('./v1/5_scale_area.jpg',blank_img)
    eroded2 = cv2.erode(_img, kernel3, iterations=1)
    eroded2 = cv2.dilate(eroded2, kernel5, iterations=2)
    cv2.imwrite('./v1/5_cut_dilate2.jpg', eroded2)
    unused, numZone_adp, hearts = findContours(eroded2, _img, hearts)
    unused, non_numZone_adp,unused = findContours(eroded2, none_pointer_img)
    cv2.imwrite('./v1/5_cut_res.jpg', numZone_adp)
    return numZone_adp,[hearts[1],hearts[2]],non_numZone_adp

def load_cnn():
    model_path = './cnn/num_cnn.h5'
    K.clear_session()  # Clear previous models from memory.
    cnn_model = load_model(model_path)
    return cnn_model



if __name__ == '__main__':

    #查找仪表圆形区域
    print('1.查找仪表圆形区域')
    cut_Img,cut_canny,cut_origin,grayImg = findMainZone('./位置2/30/image2.jpg')
    # cut_Img,cut_canny,cut_origin,grayImg = findMainZone('./位置5/image5.jpg')
    # cut_Img,cut_canny,cut_origin,grayImg = findMainZone('./image12.jpg')
    #查找指针位置
    print('2.查找指针位置')
    pointer_img = findPointer(cut_canny)
    #指针角度计算
    # calcAngle(ang1)
    # 找圆心
    print('3.找圆心')
    heartsArr,allArr = findHearts(grayImg, cut_origin)
    # 构建无指针图
    print('4.构建无指针图')
    non_pointer = getNonPointer(cut_Img,pointer_img)
    print('5.裁剪刻度区域')
    cut_Img,heartsArr,non_pointer=getScaleArea(allArr, cut_Img,non_pointer)


    print('6.提前训练训练集')
    # slearn = learnNums()
    cnn = load_cnn()
    # 第一区域获取
    print('7.第一区域获取')
    numZoneCanny,heartsArr,bigZone = getBigArea(cut_Img,heartsArr)
    # 第一区域处理，获得数字
    print('8.第一区域，处理刻度')
    bigBorder,b_pos,b_len = processBigKedu(bigZone)
    # 第一区域，处理刻度
    print('9.第一区域处理，获得数字')
    processBigNum(bigZone,bigBorder,b_pos,b_len)


    # # 第二区域
    print('10.第二区域')
    scaleShape, non_polar, non_border, non_kedu_area, nums_adp_non, horiz = processNonPointer(non_pointer,heartsArr[1])
    print('10.第二区域，处理刻度')
    s_pos,s_len =processSmallKedu(numZoneCanny,heartsArr[1],scaleShape,non_polar,non_border)
    print('11.第二区域，处理数字')
    processSmallNum(nums_adp_non,non_border,s_pos,s_len)
