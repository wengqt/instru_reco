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
        # cv2.rectangle(dst, (x-offset1, y-offset1), (x + w+offset2, y + h+offset2), (0, 255, 0), 2)
        new_img = dst[y-offset1:y + h+offset2,x-offset1:x + w+offset2]
    new_img_canny = dstCanny[y-offset1:y + h+offset2,x-offset1:x + w+offset2]
    cv2.imwrite('./findpointer/cut_findContours.jpg', new_img_canny)
    _heart= None
    if heartsarr is not None:
        _heart = heartsarr.copy()
        # print(heartsarr)
        for a in _heart:
            a[0] =a[0]-x+offset1
            a[1] = a[1] - y + offset1
    return new_img,new_img_canny,_heart





# cutImg = cv2.GaussianBlur(cutImg, (5, 5), 0)

# cutImg = cv2.equalizeHist(cutImg)
# cv2.imwrite('./findpointer/cut_enhence.jpg',cutImg)









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
# cv2.imwrite('./findpointer/line1.jpg',cutImg_1)


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
    cv2.imwrite('./findpointer/cut_find_heart.jpg',src)
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

    cv2.imwrite('./findpointer/cut_Circles.jpg',cuto_cp)
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




def get_border(arr,delta):
    '''

    :param arr: input array
    :param delta: the min range between arr item
    :return: border
    '''
    getTwo=False
    border=[]
    for i in range(len(arr)):
        if getTwo==False:
            if arr[i] !=0:
                border.append(i)
                getTwo = True
        else:
            if arr[i] ==0:
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

    line_border = []

    border_arr = get_border(horizon, min_range)


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
    non_p = cv2.subtract(cutImg, cv2.dilate(pointerImg, kernel3, iterations=3))
    cv2.imwrite('./findpointer/non_pointer.jpg', non_p)
    return non_p


def processNonPointer(non_p,one_heart):
    # unused, non_numZone_adp, unused = findContours(cut_mask, non_p, offset1=85, offset2=70)
    zoneshape = non_p.shape
    non_numZone_adp = cv2.resize(non_p, (zoneshape[1] * 8, zoneshape[0] * 8))
    cv2.imwrite('./findpointer/x_cut_non_numZoneCanny.jpg', non_numZone_adp)
    polar_ = cv2.logPolar(non_numZone_adp, (one_heart[0] * 8, one_heart[1] * 8), 600, cv2.WARP_FILL_OUTLIERS)
    polar_ = cv2.rotate(polar_, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite('./findpointer/x_cut_polar_1_non.jpg', polar_)
    unused, numsCanny_non, unused1 = findContours(cv2.dilate(polar_, kernel5, iterations=2), polar_, offset1=40,
                                                  offset2=30, big_index=0)
    numsShape = numsCanny_non.shape
    numsCanny_non = cv2.resize(numsCanny_non, (numsShape[1] * 4, numsShape[0] * 4))
    threshold, numsCanny_non = cv2.threshold(numsCanny_non, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    numsCanny_non = cv2.erode(numsCanny_non,kernel3,iterations=1)
    cv2.imwrite('./findpointer/x_nums_canny1_non.jpg', numsCanny_non)
    line_border_, ho = getLineBorder(numsCanny_non, 20)
    kedu_ = numsCanny_non[line_border_[0]:line_border_[1], :]
    # print(ho)
    return numsShape,polar_,line_border_,kedu_,numsCanny_non,ho


# def processSmallKedu(zone2polar,one_heart,numsShape,polar_,line_border_):
#     polar = cv2.logPolar(zone2polar, (one_heart[0]*8, one_heart[1]*8), 600, cv2.WARP_FILL_OUTLIERS)
#     polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cv2.imwrite('./findpointer/x_cut_polar_1.jpg',polar)
#     unused,numsCanny_1,unused1=findContours(cv2.dilate(polar_,kernel5,iterations=2),polar,offset1=40,offset2=30,big_index=0)
#     numsCanny_1 = cv2.resize(numsCanny_1,(numsShape[1]*4,numsShape[0]*4))
#     cv2.imwrite('./findpointer/x_nums_canny1.jpg',numsCanny_1)
#     kedu_1_pointer = numsCanny_1[line_border_[0]:line_border_[1]+60,:]
#     kedu_1_pointer = cv2.erode(kedu_1_pointer,kernel3)
#     cv2.imwrite('./findpointer/x_nums_kedu_1_pointer.jpg',kedu_1_pointer)
#     ver_ = projectVertical(kedu_1_pointer)
#     (h1, w1) = kedu_1_pointer.shape
#     newHorizon_ = np.zeros([h1, w1], np.uint8)
#
#     for i in range(0, w1):
#         for j in range(0, ver_[i]):
#             newHorizon_[j, i] = 255
#
#     cv2.imwrite('./findpointer/x_nums_kedu_111.jpg',newHorizon_)
#     maxtab,mintab = peakdetective.peakdet(ver_,30)
#     # from matplotlib.pyplot import plot, scatter, show
#     # plot(ver)
#     # scatter(np.array(maxtab)[:, 0], np.array(maxtab)[:, 1], color='blue')
#     # scatter(np.array(mintab)[:, 0], np.array(mintab)[:, 1], color='red')
#     # show()
#     k_res = list(maxtab[:,1])
#     # print(res.index(max(res)),len(res))
#     k_pos =  k_res.index(max(k_res))
#     k_len = len(k_res)
#     print('position',k_res.index(max(k_res)),'总刻度线数：',(len(k_res)-1))
#     return k_pos,k_len

#处理数字
# def processSmallNum(numsCanny_non,line_border_,k_pos,k_len):
#     numsimg_ = numsCanny_non[line_border_[2]:line_border_[3],:]
#     numsimg_ = cv2.erode(numsimg_,kernel5)
#
#     numsdilate_ = cv2.dilate(numsimg_,kernel5,iterations=6)
#     cv2.imwrite('./v3/xi_numsimg_.jpg',numsdilate_)
#     numsArr_ = getEachNum(numsdilate_,60)[0]
#
#
#     nums_ = cutNums(numsArr_,numsimg_,'./v3/nums1/')
#     index_=0
#
#     numbers_ =[]
#     for one_num in nums_:
#         num_border = getEachNum(one_num)[0]
#         tmpArr = []
#         for i in range(0,len(num_border),2):
#             # print(num_border[i],num_border[i+1])
#             new_cut = one_num[:,num_border[i]:num_border[i+1]]
#             new_cut=cv2.resize(new_cut,(64,128))
#             cv2.imwrite('./v3/nums1/' +str(index_)+'_'+ str(int(i / 2)) + '.jpg', new_cut)
#             # new_cut = Image.fromarray(cv2.cvtColor(new_cut, cv2.cv2.COLOR_BGR2GRAY))
#             hight, width = new_cut.shape
#             new_cut = np.asarray(new_cut)
#             # new_cut = new_cut.reshape(1, hight * width)[0]
#
#             tmpArr.append(new_cut)
#         index_ += 1
#         numbers_.append(tmpArr)
#
#
#
#
#     testDigits = [numbers_[0],numbers_[len(numbers_)-1]]
#
#     kedu_range = []
#     for test in testDigits:
#         res = ''
#         for i in test:
#
#
#             x = cv2.resize(i,(32,64))
#             x = x.astype('float32')
#             x /= 255
#             x = x.reshape(1, 64, 32, 1)
#             pr = cnn.predict(x)
#             pr = convert2Num(pr)
#             # convert2Num(pr)
#             print('cnn识别结果：',pr)
#             # i = i.astype('float32')
#             # i /= 255
#             # i = i.reshape(1, 128 * 64)[0]
#             # predict = slearn.predict(i)
#
#             res += '-' if str(pr)=='10' else str(pr)
#         print(res)
#         kedu_range.append(int(res))
#
#     result = k_pos*(k_len-1)/(kedu_range[1]-kedu_range[0])+kedu_range[0]
#     print('结果：',result)

def learnNums():
    softmax_learn = softmax.Softmax()
    trainDigits, trainLabels = softmax_learn.loadData('./train')
    softmax_learn.train(trainDigits, trainLabels, maxIter=100)  # 训练
    return softmax_learn


def findMainZone(path):
    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (3000, 2000))
    # canny = cv2.Canny(cv2.GaussianBlur(img1, (7, 7), 0), 100, 250)
    # cv2.imwrite('./v3/canny.jpg', canny)

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
    cv2.imwrite('./v3/1_circles.jpg', img1_cp)

    cutImg_o = img1[the_circle[1] - the_circle[2]:the_circle[1] + the_circle[2],
               the_circle[0] - the_circle[2]:the_circle[0] + the_circle[2]]
    cv2.imwrite('./v3/1_cut.jpg', cutImg_o)

    #对裁剪的图片进行二值化和平滑处理。
    cutImg = cv2.cvtColor(cutImg_o, cv2.COLOR_RGB2GRAY)
    cut_g = cv2.GaussianBlur(cutImg, (7, 7), 1)
    cut_g = cv2.equalizeHist(cut_g)
    cutCanny = cv2.Canny(cut_g, 50, 100)
    cv2.imwrite('./v3/1_cut_Canny.jpg', cutCanny)
    # kernel = np.ones((3, 3), np.float32) / 25
    # cutImg1 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg1 = cv2.bilateralFilter(cutImg, 9, 70, 70)
    # cutImg1 = cv2.GaussianBlur(cutImg,(7,7),0) #灰度图，用于后面的圆圈识别
    kernel = np.ones((3, 3), np.float32) / 25
    cutImg2 = cv2.filter2D(cutImg, -1, kernel) #灰度图，用于后面的圆圈识别
    cutImg3 = cv2.adaptiveThreshold(cutImg2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
    cv2.imwrite('./v3/1_cut_adapt.jpg', cutImg3)
    return cutImg3,cutCanny,cutImg_o,cutImg1



def findPointer2(_img,_heart):
    '''

    :param _img: 二值图
    :param _heart: 两个刻度盘的圆心
    :return: 无指针图
    '''
    # _img = cv2.resize(_img,(500,500))
    _shape = _img.shape
    _img1 = _img.copy()
    _img1 = cv2.erode(_img1, kernel3, iterations=1)
    # _img1 = cv2.dilate(_img1, kernel3, iterations=1)
    _count = 0
    _imgarr=[]
    for item in _heart:

        #157=pi/2*100
        mask_max = 0
        mask_theta = 0
        for i in range(0,314):
            black_img = np.zeros([_shape[0], _shape[1]], np.uint8)
            theta = float(i)*0.01
            y1 = int(item[0]+math.cos(theta+np.pi/2)*_shape[0])
            x1 = int(item[1]+math.sin(theta+np.pi/2)*_shape[0])
            # cv2.circle(black_img, (x1, y1), 2, 255, 3)
            cv2.line(black_img, (item[0], item[1]), (x1, y1), 255, 3)

            tmp = np.mean(cv2.bitwise_and(black_img,_img1))
            if tmp>mask_max:
                mask_max=tmp
                mask_theta=theta
            # cv2.imwrite('./v3/2_line1.jpg', black_img)

        black_img = np.zeros([_shape[0], _shape[1]], np.uint8)
        y1 = int(item[0] + math.cos(mask_theta + np.pi / 2) * _shape[0])
        x1 = int(item[1] + math.sin(mask_theta + np.pi / 2) * _shape[0])
        cv2.line(black_img, (item[0], item[1]), (x1, y1), 255, 7)
        cv2.imwrite('./v3/2_theta'+str(_count)+'.jpg',black_img)

        # black_img1 = np.zeros([_shape[0], _shape[1]], np.uint8)
        # y1 = int(item[0] + math.cos(mask_theta + np.pi / 2) * 1000)
        # x1 = int(item[1] + math.sin(mask_theta + np.pi / 2) * 1000)
        # cv2.line(black_img1, (item[0], item[1]), (x1, y1), 255, 6)

        _img = cv2.subtract(_img,black_img)
        _img1 = cv2.subtract(_img1,black_img)
        # cv2.imwrite('./v3/2_theta__' + str(_count) + '.jpg', _img1)
        _imgarr.append(_img)
        _count +=1

    cv2.imwrite('./v3/2_non_pointer.jpg', _imgarr[1])
    return _imgarr











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
    # cv2.imwrite('./v3/6_cut_numZoneCanny.jpg', zone)
    M=zoneshape[1] * 4/math.log(_heart[2]*4)
    print(M)
    # 极坐标转换
    polar = cv2.logPolar(zone, (_heart[0] * 8, _heart[1] * 8), M, cv2.WARP_FILL_OUTLIERS)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if type==2:

        zone2 = cv2.resize(zone2, (zoneshape[1] * 8, zoneshape[0] * 8))
        polar2 = cv2.logPolar(zone2, (_heart[0] * 8, _heart[1] * 8), M, cv2.WARP_FILL_OUTLIERS)
        polar2 = cv2.rotate(polar2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        non_area, area, unused1 = findContours(cv2.dilate(polar, kernel5, iterations=1), polar, dst=polar2)
        # print(non_area)

    else:
        unused, area, unused1 = findContours(cv2.dilate(polar, kernel5, iterations=1), polar)

    cv2.imwrite('./v3/6_cut_polar.jpg', polar)


    numsShape = area.shape
    numsCanny = cv2.resize(area, (numsShape[1] * 4, numsShape[0] * 4))

    threshold, area = cv2.threshold(numsCanny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('./v3/6_nums_canny.jpg', area)  # 直着的 带刻度带数字图



    if type==1:

        border, ho = getLineBorder(area, 20)
        _kedu = area[border[2]:, :]
        _num = area[border[0]:border[1],:]
    else:
        non_numsCanny = cv2.resize(non_area, (numsShape[1] * 4, numsShape[0] * 4))
        threshold, non_area = cv2.threshold(non_numsCanny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border, ho = getLineBorder(non_area, 20)
        cv2.imwrite('./v3/6_nums_canny.jpg', non_area)  # 直着的 带刻度带数字图
        _num = area[border[2]:border[3], :]
        _kedu = area[border[0]:, :].copy()
        _kedu[border[2]-border[0]:border[3]-border[0], :]=0

    cv2.imwrite('./v3/6_nums_'+str(type)+'.jpg', _num)  # 直着的 带刻度带数字图
    cv2.imwrite('./v3/6_kedu_'+str(type)+'.jpg', _kedu)  # 直着的 带刻度带数字图
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


def processNum(numarea,k_pos,k_len,index):

    numsimg = numarea
    numsdilate = cv2.erode(numsimg, kernel4, iterations=2)
    numsdilate = cv2.dilate(numsdilate, kernel5, iterations=3)
    cv2.imwrite('./v3/9_numsdilate.jpg', numsdilate)
    numsArr = getEachNum(numsdilate)[0]
    nums = cutNums(numsArr, numsimg, './v3/num'+str(index)+'/')
    _index = 0
    numbers_ = []
    for one_num in nums:
        num_border = getEachNum(one_num)[0]
        tmpArr = []
        res = ''
        for i in range(0, len(num_border), 2):
            new_cut = one_num[:, num_border[i]:num_border[i + 1]]
            new_cut = cv2.resize(new_cut, (32, 64))
            cv2.imwrite('./v3/num'+str(index)+'/' + str(_index) + '_' + str(int(i / 2)) + '.jpg', new_cut)

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
            # x = np.array([x])
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
        kedu_range.append(int(res))

    result = k_pos * (k_len - 1) / (kedu_range[1] - kedu_range[0]) + kedu_range[0]
    print('结果：',result)




def processKedu(zone,index):

    ver = projectVertical(zone)
    (h1, w1) = zone.shape
    newHorizon = np.zeros([h1, w1], np.uint8)
    for i in range(0, w1):
        for j in range(0, ver[i]):
            newHorizon[j, i] = 255
    cv2.imwrite('./v3/7_nums_kedu_'+str(index)+'.jpg', newHorizon)
    maxtab, mintab = peakdetective.peakdet(ver, 3)

    k_res = list(maxtab[:, 1])
    # print(res.index(max(res)),len(res))
    k_pos = k_res.index(max(k_res))
    k_len = len(k_res)
    print('指针位置',k_res.index(max(k_res)), '总刻度线个数',len(k_res))
    # print('position', k_res.index(max(k_res)) / (len(k_res) - 1) * 100)

    return k_pos,k_len



def getScaleArea(heart_arr,_img,non_img_1,_non):
    '''

    :param heart_arr: 圆心数组
    :param _img: 表盘二值图
    :param p_img: 无指针图
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
    cv2.imwrite('./v3/5_area1.jpg', _img1)

    _img2 = cv2.bitwise_and(blank_img2, non_img_1)
    cv2.imwrite('./v3/5_area2.jpg', _img2)


    _non1 = cv2.bitwise_and(_non, blank_img1)
    _non2 = cv2.bitwise_and(_non, blank_img2)

    eroded1 = cv2.erode(_img1, kernel3, iterations=1)
    eroded1 = cv2.dilate(eroded1, kernel5, iterations=2)
    cv2.imwrite('./v3/5_cut_dilate1.jpg', eroded1)
    eroded2 = cv2.erode(_img2, kernel3, iterations=1)
    eroded2 = cv2.dilate(eroded2, kernel5, iterations=2)
    cv2.imwrite('./v3/5_cut_dilate2.jpg', eroded2)

    _hearts= hearts.copy()

    cut_non1, _cutnumZone1, hearts1 = findContours(eroded1, _img1, _hearts,_non1)

    cut_non2, _cutnumZone2, hearts2 = findContours(eroded2, _img2, _hearts,_non2)
    # unused, non_numZone_adp,unused = findContours(eroded2, none_pointer_img)
    cv2.imwrite('./v3/5_cut_res1.jpg', _cutnumZone1)
    cv2.imwrite('./v3/5_cut_res2.jpg', _cutnumZone2)
    cv2.imwrite('./v3/5_cut_p_img1.jpg', cut_non1)
    cv2.imwrite('./v3/5_cut_p_img2.jpg', cut_non2)
    return _cutnumZone1,_cutnumZone2,hearts1[1],hearts2[2],cut_non1,cut_non2

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

    # 找圆心
    print('3.找圆心')
    heartsArr = findHearts(grayImg, cut_origin)

    #查找指针位置
    print('2.查找指针位置')
    # pointer_img = findPointer(cut_canny)
    # pointer_img = findPointer(cut_Img,heartsArr)
    # pointer_img = findPointer(gray2,heartsArr)
    #指针角度计算
    # calcAngle(ang1)
    non_img_arr = findPointer2(cut_Img, heartsArr[1:3])



    print('5.裁剪刻度区域')
    cut_Img1,cut_Img2,heart1,heart2,non_img1,non_img2=getScaleArea(heartsArr, cut_Img,non_img_arr[0],non_img_arr[1])

    # print(heart1,heart2)
    print('6.进行极坐标转换')
    kedu1,num1 = convertPolar(cut_Img1, heart1,1)
    kedu2,num2 = convertPolar(cut_Img2, heart2,2,non_img2)
    #
    print('提前加载模型')
    cnn = load_cnn()

    print('7.第一区域kedu处理')
    pos1,len1=processKedu(kedu1,index=1)

    print('8.第一区域数字处理')
    processNum(num1, pos1, len1, index=1)

    print('9.第二区域kedu处理')
    pos2, len2 = processKedu(kedu2,index=2)

    print('10.第二区域数字处理')
    processNum(num2, pos2, len2,index=2)



