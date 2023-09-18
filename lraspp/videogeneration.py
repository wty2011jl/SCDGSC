import numpy as np
import cv2
from predict import main
#读取一张图片
size = (693,520) #first_frame.shape 的输出是[高，宽，通道数]，而 frameSize=[宽，高]，所以需要调换顺序
# print(size)
#完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
fourcc = cv2.VideoWriter_fourcc(*'h264')
videowrite = cv2.VideoWriter(r'C:\Users\12955\Desktop\change_driven SemCom\Video_Generation\SEMtest.avi',fourcc,20,size)#20是帧数，size是图片尺寸
img_array=[]
for i in range(1231):
    print("i", i)
    main(i)
    img = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Video_Generation\results1\1_%d.png" % i)
    img_array.append(img)
for i in range(1231):
    videowrite.write(img_array[i])
print('end!')