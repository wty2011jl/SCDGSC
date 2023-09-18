import cv2
import torch
from PIL import Image
import numpy as np
from utils import postprocess_output
from torchvision import transforms
# sunny = cv2.imread(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\reference\1.png")
# for i in range(1231):
#     b = cv2.imread( r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\reference\1_%d.jpg" % i, cv2.IMREAD_GRAYSCALE)
#     #ret, b = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY)
#     print("b", b.shape)
    #cv2.imwrite(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\segmap_image\1_%d.png" % i, b)

# rainy = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\ref\2.jpg")
# for i in range(1231):
#     cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\reference_image\2_%d.jpg" % i, rainy)
#
# snowy = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\ref\3.jpg")
# for i in range(1231):
#     cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\reference_image\3_%d.jpg" % i, snowy)
#
# haze = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\ref\4.jpg")
# for i in range(1231):
#     cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\reference_image\4_%d.jpg" % i, haze)

# for i in range(1,5):
#     for j in range(1231):
#         a = cv2.imread(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset\training_image\%d_%d.jpg" % (i,j))
#         c = cv2.resize(a, (128,96))
#         #print("a", a)
#         cv2.imwrite(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\training_image\%d_%d.jpg" % (i,j), c)
#         a = cv2.imread(
#             r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset\reference_image\%d_%d.jpg" % (i, j))
#         c = cv2.resize(a, (128,96))
#         # print("a", a)
#         cv2.imwrite(
#             r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\reference_image\%d_%d.jpg" % (i, j),
#             c)
#         a = cv2.imread(
#             r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset\segmap_image\%d_%d.png" % (i, j),0)
#         c = cv2.resize(a, (128,96))
#         # print("a", a)
#         cv2.imwrite(
#             r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\segmap_image\%d_%d.png" % (i, j),
#             c)

# save_path = r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\results\222.jpg"
# x = torch.randn(3, 240, 320)
# x = postprocess_output(x).data.numpy().transpose(1, 2, 0)
# Image.fromarray(np.uint8(x)).save(save_path)

# data_transform = {
#     "three": transforms.Compose([transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#     "one": transforms.Compose([transforms.ToTensor(),
#                                transforms.Normalize(0.485, 0.229)])}


# class UnNormalize:
#     #restore from T.Normalize
#     #反归一化
#     def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
#         self.mean=torch.tensor(mean).view((1,-1,1,1))
#         self.std=torch.tensor(std).view((1,-1,1,1))
#     def __call__(self,x):
#         x=(x*self.std)+self.mean
#         return torch.clip(x,0,None)
#
# TT_1 = data_transform["three"]
# TT_2 = data_transform["one"]
# UTT_1 = UnNormalize(mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225))
# UTT_2 = UnNormalize(mean= 0.485, std = 0.229)
#
# a = Image.open(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\reference_image\1_71.jpg")
# c = np.asarray(a)
# print("aa", c, c.shape)
# a = TT_1(a)
# print("a",a, a.shape)
# b = UTT_1(a)
# print("b",b, b.shape)

# b = cv2.imread(r"C:\Users\user\Desktop\change_driven SemCom\VOC_CGSC\SegmentationClass\1_320.png")
# b = b * 128
# cv2.imshow("a",b)
# cv2.waitKey(0)

#        cv2.imwrite(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\training\2_%d.jpg" % j, c)

b = cv2.imread(r"C:\Users\user\Desktop\change_driven SemCom\Video_Generation\results1\1_1057.png",
        cv2.IMREAD_GRAYSCALE)
ret, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY)
c = cv2.resize(b, (128,96))
        #print("a", a)
cv2.imwrite(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\denoise.jpg", c)