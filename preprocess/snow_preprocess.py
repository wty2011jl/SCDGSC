#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     : 2022.11
# @Author   : 绿色羽毛
# @Email    : lvseyumao@foxmail.com
# @Blog     : https://blog.csdn.net/ViatorSun
# @Paper    :
# @arXiv    :
# @version  : "1.0"
# @Note     :
#


import cv2
import random
import numpy as np


img = cv2.imread('IMG_7016.jpg')


# 运动模糊
def motion_blur(img, degree=15, angle=30):
    M = cv2.getRotationMatrix2D((degree/2,degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))

    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel,M ,(degree,degree))
    motion_blur_kernel = motion_blur_kernel / degree

    blurred = cv2.filter2D(img, -1, motion_blur_kernel)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred




def snow(img):
    bg = np.zeros_like(img, dtype='uint8')
    snow_list = []
    Num = random.randint(300,600)
    for i in range(Num):
        w,h,_ = bg.shape
        x = random.randrange(0, h)
        y = random.randrange(0, w)
        sx = random.uniform(-1, 1)
        sy = random.randint(1,2)
        snow_list.append([x, y, sx, sy])

    # 雪花列表循环
    for i in range(len(snow_list)):
        # 绘制雪花，颜色、位置、大小
        xi = snow_list[i][0]
        yi = snow_list[i][1]
        cv2.circle(bg, (xi, yi), snow_list[i][3], thickness=-1, color=(255,255,255))

    return bg

for i in range(0,1231):
    print("i", i)
    img = cv2.imread(r'C:\Users\12955\Desktop\change_driven SemCom\dataset_vehicle\1 highway\t1\1_%d.jpg' %i)
    bg = snow(img)
    #bg = motion_blur(bg)
    img = cv2.add(img, bg)
    cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\JPEGImages\3_%d.jpg" % i, img)




# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
