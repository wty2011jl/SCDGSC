
import cv2
import os
# 文件夹完整路径
# wjj = r"C:\Users\12955\Desktop\change_driven SemCom\dataset_vehicle\4 haze_highway\t1"
# wii = r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\JPEGImages"
# # 循环遍历文件夹中所有文件，获取文件名及编号
# for n, name in enumerate(os.listdir(wjj)):
#     # 原文件的路径及名称
#     src = wjj + "/" + name
#
#     # 重命名后文件路径及名称
#     #dst = wii + "/" + "3" + name[1:] + ".jpg"
#     dst = wii + "/" + "4_" + str(n) + ".jpg"
#     # 重命名文件
#     os.rename(src, dst)

# for i in range(0,200):
#     print("i", i)
#     kk = i + 800
#     img = cv2.imread(r'C:\Users\12955\OneDrive\cd2014_part1\lowFramerate\turnpike_0_5fps\groundtruth\gt000%d.png' %kk)
#     #src = r'C:\Users\12955\Desktop\change_driven SemCom\dataset_vehicle\9 streetLight\t1\in000%d.jpg' %kk
#     #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\SegmentationClass\8_%d.png" % i, img)
#     #dst = r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\JPEGImages\5_%d.jpg" % i
#     #os.rename(src, dst)
# # img = cv2.imread(r'C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\SegmentationClass\1_0.png')
# # print (img.shap

# wjj = r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\SegmentationClass"
# for n, name in enumerate(os.listdir(wjj)):
#     print("n",n)
#     img = cv2.imread(wjj+'/'+name)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     #r, img = cv2.threshold(img,0,1,cv2.THRESH_BINARY)
#     #print(img.shape)
#     #2.waitKey(0)
#     cv2.imwrite(wjj+'/'+name, img)
    #print(img.shape)

# for i in range(1,5):
#     for j in range (1231):
#         img = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\training_image\%d_%d.jpg" % (i,j))
#         c = cv2.resize(img, (32, 24))
#         cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\training_image\%d_%d.jpg" % (i,j), c)
#
# for i in range(1,5):
#     for j in range (1231):
#         img = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\reference_image\%d_%d.jpg" % (i,j))
#         c = cv2.resize(img, (32, 24))
#         cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\reference_image\%d_%d.jpg" % (i,j), c)

for i in range(1,5):
    for j in range (1231):
        img = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\segmap_image\%d_%d.png" % (i,j),0)
        c = cv2.resize(img, (32, 24))
        cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\segmap_image\%d_%d.png" % (i,j), c)

