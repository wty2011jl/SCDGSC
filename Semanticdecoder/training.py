import random
#import imageio
import numpy as np
from argparse import ArgumentParser
from ddpmmodel import MyDDPM
from unetmodel import UNet
from training_loop import training_loop
from utils import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torchvision import transforms

from mydataset import MyDataSet
from utils import read_split_data, plot_data_loader_image
#import einops # gif visualization
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda #preprocessing
#from torchvision.datasets.mnist import  MNIST, FashioMNIST


Init_lr             = 2e-5
Min_lr              = Init_lr * 0.01
momentum            = 0.9
weight_decay        = 0
# root1 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\training"  # 数据集所在根目录
# root2 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\segmap_image"
# root3 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\reference"
# root1 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_one_reference\training_image"
# root2 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_one_reference\segmap_image"
# root3 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_one_reference\reference_image"

# root1 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_third_reference\training_image"
# root2 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_third_reference\segmap_image"
# root3 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_third_reference\reference_image"

root1 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_forth_reference\training_image"
root2 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_forth_reference\segmap_image"
root3 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_forth_reference\reference_image"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

train_images_path, segmap_images_path, ref_images_path = read_split_data(root_training=root1,root_segmap=root2,root_ref=root3)

data_transform = {
    "three": transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "one": transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(0.485, 0.229)])}
# TT = data_transform["one"]
#
# K = Image.open(r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset\segmap_image\1_0.png")
# print("dddddd", K)
# S = TT(K)
# print("ll", S)
nn = len(train_images_path)
print(nn)
batch_size = 6
train_data_set = MyDataSet(train_images_path, segmap_images_path, ref_images_path,
                           transform_three=data_transform["three"], transform_one = data_transform["one"])
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers'.format(nw))
train_loader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=train_data_set.collate_fn)
train_flag = True
loader = train_loader
print("dddd",loader)
n_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("s",device)
#pretrain_model_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\decoder -0-200\end_ddpm_model_one_reference_s_4_0_200epoch.pt"
#weights_dict = torch.load(pretrain_model_path, map_location='cpu')
MyUNet = UNet(3).to(device)
ddpm = MyDDPM(MyUNet,n_steps = 1000, min_beta = 10 ** (-4), max_bata = 0.02, device = device, image_chw=(3, 128, 96))
#ddpm.load_state_dict(weights_dict)
ddpm.to(device)
ddpm_train = ddpm.train()
optim = torch.optim.Adam(ddpm_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay)
#print("aaass", ddpm.alphas_bars.device)

if train_flag:
    training_loop(ddpm_train, loader, n_epochs, optim=optim, device=device)

#Testing and Generation

# best_model = MyDDPM(MyUNet, n_steps = 200, min_beta = 10 ** (-4), max_bata = 0.02, device = None, image_chw=(1, 28, 28)) #network structure
# best_model.load_state_dict(torch.load(store_path, map_location=device)) #加载模型参数字典
# best_model.eval()
# print("Model loaded")
#
# generated =  generate_new_images(
#     best_model,
#     n_samples = 100,
#     device = deivce,
#     gif_name = "fashion.gif" if fashion else "mnist.gif"
# )
#
# show_images(generated, "Final result")