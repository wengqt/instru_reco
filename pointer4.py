import cv2
import numpy as np
import sys
import math
import peakdetective
import matplotlib.pyplot as plt
# from keras.models import load_model
# from keras import backend as K


kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))


def findMainZone(path):
    img1 = cv2.imread(path)
    try:
        if img1 is None:
            raise Exception('找不到图片! 图片路径有误。')
    except Exception as err:
        # print(err)
        print('err', err)
        sys.exit(1)

    # src = cv2.pyrMeanShiftFiltering(img1, 500, 10)
    src = cv2.resize(cv2.bilateralFilter(img1,11, 90, 90),(int(img1.shape[1]/4),int(img1.shape[0]/4)))
    # src = cv2.resize(img1,(int(img1.shape[1]/4),int(img1.shape[0]/4)))

    # src = cv2.medianBlur(src, 9)
    img_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)


    canny = cv2.Canny(img_gray,180,200)
    cv2.imwrite('./pointer4/0_gray.jpg', canny)

    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 60, minLineLength= minLineLength,maxLineGap=maxLineGap)
    # print(lines[0])

    ver_lines=[]
    hor_lines=[]
    delta = 7
    for l in lines:
        for x1, y1, x2, y2 in l:

            if abs(x1-x2)< delta:
                ver_lines.append([x1, y1, x2, y2])
                cv2.line(canny, (x1, y1), (x2, y2), 255, 20)
            elif abs(y1-y2)< delta:
                hor_lines.append([x1, y1, x2, y2])
                cv2.line(canny, (x1, y1), (x2, y2), 255, 20)
    ef_lines = np.array(ver_lines+hor_lines)
    # print(ef_lines[:,0])
    center_x = int((sum(ef_lines[:,0])+sum(ef_lines[:,2]))/(2*len(ef_lines)))
    center_y = int((sum(ef_lines[:,1])+sum(ef_lines[:,3]))/(2*len(ef_lines)))
    print((center_x, center_y))
    cv2.circle(canny, (center_x,center_y), 20, 255, -1)
    cv2.imwrite('./pointer4/0_gray.jpg', canny)
    left, right, down, up = classify_lines(ver_lines,hor_lines,center_x,center_y)
    # x1 = fitting_line(canny,left,1)
    # x2 = fitting_line(canny,right,1)
    # y1 = fitting_line(canny,up,0)
    # y2 = fitting_line(canny,down,0)
    #
    # cut = src[int(y1):int(y2),int(x1):int(x2)]
    # cv2.imwrite('./pointer4/0_cut.jpg', cut)
    x1 = calc_lines(canny,left,center_x,center_y,1)
    x2 = calc_lines(canny,right,center_x,center_y,1)
    y1 = calc_lines(canny,up,center_x,center_y,0)
    y2 = calc_lines(canny,down,center_x,center_y,0)

def fitting_line(img,arr,type_):
    '''

    :param img:
    :param arr:
    :param type_: 1 for vertical, 0 for horizon
    :return:
    '''
    mask = np.zeros(img.shape,np.uint8)
    for [x1, y1, x2, y2] in arr:
        # for x1, y1, x2, y2 in l:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 20)

    if type_==1:
        mask = cv2.dilate(mask,kernel2,iterations=4)
    else:
        mask = cv2.dilate(mask, kernel1,iterations=4)

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)

    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    # print(lefty)
    cv2.circle(img, (x, y), 20, 255, -1)
    cv2.line(img, (cols - 1, righty), (0, lefty), 255, 2)

    cv2.imwrite('./pointer4/0_mask.jpg', img)
    if type_==0:
        return y
    elif type_==1:
        return x



def calc_lines(img,arr,cx,cy,type_):
    # arr = np.array(arr)
    mid_arr = []
    if type_ ==1:
        for [x1, y1, x2, y2] in arr:
            mid_arr.append((x1+x2)/2)
        mid_arr = sorted(mid_arr, key=lambda x: abs(cx - x), reverse=True)
    else:
        for [x1, y1, x2, y2] in arr:
            mid_arr.append((y1+y2)/2)
        mid_arr = sorted(mid_arr, key=lambda x: abs(cy - x), reverse=True)


    sum =0
    sum_i = 0
    for i in range(len(mid_arr)):
        sum += (i+1)*mid_arr[i]
        sum_i += i+1

    if sum_i == 0:
        sum_i = 1
    avg = int(sum/sum_i)
    # print(sum,sum_i,avg)
    if type_==1:
        img[:,avg] = 255
    else:
        img[avg,:] = 255
    cv2.imwrite('./pointer4/0_mask.jpg', img)
    return avg





def classify_lines(vlines,hlines,cx,cy):
    left= []
    right= []
    down= []
    up= []
    print(vlines)
    for [x1, y1, x2, y2] in vlines:
        # for  in l:
            if x1<cx and x2<cx:
                left.append([x1, y1, x2, y2])
            elif x1>cx and x2>cx:
                right.append([x1, y1, x2, y2])


    for [x1, y1, x2, y2] in hlines:
        # for x1, y1, x2, y2 in l:
            if y1<cy and y2<cy:
                up.append([x1, y1, x2, y2])
            elif y1>cy and y2>cy:
                down.append([x1, y1, x2, y2])



    return left, right, down, up



def calc_line_gray(gray_src,arr,type_):
    '''

    :param gray_src:
    :param arr:
    :param type_:0 up,1 left,2 down,3 right
    :return:
    '''



    return arr







if __name__ == '__main__':
    pth = './img/image1.jpg'
    findMainZone(pth)