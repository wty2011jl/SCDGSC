import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from PIL import Image

#spade norm
class SPADEGroupNorm(nn.Module):
    def __init__(self, num_group, norm_nc, label_nc = 1, eps = 1e-5): # norm_nc: the channel of input feature
        # label_nc: the channel of semantic map
        super().__init__()

        self.norm = nn.GroupNorm(num_group, norm_nc, affine=False) # 32/16

        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap=None):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        if segmap != None:
        # Part 2. produce scaling and bias conditioned on semantic map
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
            actv = self.mlp_shared(segmap)
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)
            x = x * (1 + gamma) + beta

        # apply scale and bias
        return x


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# def get_norm(norm, num_channels, num_groups):    #归一化方式
#     if norm == "in":
#         return nn.InstanceNorm2d(num_channels, affine=True)
#     elif norm == "bn":
#         return nn.BatchNorm2d(num_channels)
#     elif norm == "gn":
#         return nn.GroupNorm(num_groups, num_channels)
#     elif norm == "spade":
#         return SPADEGroupNorm(num_groups, num_channels)
#     elif norm is None:
#         return nn.Identity()
#     else:
#         raise ValueError("unknown normalization type")

class PositionalEmbedding(nn.Module):    #Scalar to Vector
    def __init__(self, dim, scale=1.0):   #dim = base_channel; scale = time_emb_scale
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):   #t time_step
        device      = x.device
        half_dim    = self.dim // 2   #整除   #一半sin;一半cos
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x * self.scale和emb外积
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y=None):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),  # kernel_size = 3
        )

    def forward(self, x, time_emb, y=None):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = nn.GroupNorm(  num_groups, in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, t = None, y = None):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class Encoder_ResidualBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None, activation=SiLU(),
             num_groups=32, use_attention=False,
    ):
        super().__init__()
        self.activation = activation

        self.norm_1 = nn.GroupNorm(  num_groups, in_channels,)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = nn.GroupNorm(  num_groups, out_channels,)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        #self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels,  num_groups)

    def forward(self, x, time_emb, y=None):
        out = self.activation(self.norm_1(x))
        # 第一个卷积
        out = self.conv_1(out)

        # 对时间time_emb做一个全连接，施加在通道上
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        # # 对种类y_emb做一个全连接，施加在通道上
        # if self.class_bias is not None:
        #     if y is None:
        #         raise ValueError("class conditioning was specified but y is not passed")
        #
        #     out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        # 第二个卷积+残差边
        out = self.conv_2(out) + self.residual_connection(x)
        # 最后做个Attention
        out = self.attention(out)
        return out


class Decoder_ResidualBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None, activation=SiLU(), num_groups=32, use_attention=False,
    ):
        super().__init__()
        self.activation = activation

        self.norm_1 = SPADEGroupNorm(  num_groups, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = SPADEGroupNorm( num_groups, out_channels)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels,  num_groups)

    def forward(self, x, time_emb=None, segmap=None):

        out = self.activation(self.norm_1(x,segmap))
        # 第一个卷积
        out = self.conv_1(out)

        # 对时间time_emb做一个全连接，施加在通道上
        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        # 对种类y_emb做一个全连接，施加在通道上
        # if self.class_bias is not None:
        #     if y is None:
        #         raise ValueError("class conditioning was specified but y is not passed")
        #
        #     out += self.class_bias(y)[:, :, None, None]
        if segmap == None:
            out = self.activation(self.norm_2(out))
        else:
            out = self.activation(self.norm_2(out,segmap))
        # 第二个卷积+残差边
        out = self.conv_2(out) + self.residual_connection(x)
        # 最后做个Attention
        out = self.attention(out)
        return out


class UNet_1(nn.Module): #x|y
    def __init__(
            self, img_channels, base_channels=64, channel_mults=(1, 2, 4, 8), #输入图片太小 否则（1，2，4，8）
            num_res_blocks=2, time_emb_dim=64 * 4, time_emb_scale=1.0, num_classes=None, activation=SiLU(),
            dropout=0.1, attention_resolutions=(1,),  num_groups=4, initial_pad=0,
    ):
        super().__init__()
        # 使用到的激活函数，一般为SILU
        self.activation = activation
        # 是否对输入进行padding
        self.initial_pad = initial_pad
        # 需要去区分的类别数
        self.num_classes = num_classes

        # 对时间轴输入的全连接层
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None

        # 对输入图片的第一个卷积
        self.init_conv = nn.Conv2d(img_channels * 2, base_channels, 3, padding=1)

        # self.downs用于存储下采样用到的层，首先利用ResidualBlock提取特征
        # 然后利用Downsample降低特征图的高宽
        self.downs = nn.ModuleList()  # 一系列下采样
        self.ups = nn.ModuleList()  # 一系列上采样

        # channels指的是每一个模块处理后的通道数
        # now_channels是一个中间变量，代表中间的通道数
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    Encoder_ResidualBlock(
                        now_channels, out_channels, dropout,
                        time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                         num_groups=num_groups, use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # 可以看作是特征整合，中间的一个特征提取模块
        self.mid = nn.ModuleList(
            [
                Encoder_ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                     num_groups=num_groups, use_attention=True,
                ),
                Decoder_ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                     num_groups=num_groups, use_attention=False,
                ),
            ]
        )
        # 进行上采样，进行特征融合
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(Decoder_ResidualBlock(
                    channels.pop() + now_channels, out_channels, dropout,
                    time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                     num_groups=num_groups, use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        self.out_norm = nn.GroupNorm(  num_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x, time=None, segmap=None):
        # 是否对输入进行padding
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        # 对时间轴输入的全连接层
        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            time_emb = self.time_mlp(time)
        else:
            time_emb = None

        # if self.num_classes is not None and y is None:
        #     raise ValueError("class conditioning was specified but y is not passed")

        # 对输入图片的第一个卷积

        x = self.init_conv(x)

        # skips用于存放下采样的中间层
        skips = [x]
        for layer in self.downs:
            #print("layer",layer)
            x = layer(x, time_emb,segmap)
            skips.append(x)

        # 特征整合与提取
        for layer in self.mid:
            x = layer(x, time_emb,segmap)

        # 上采样并进行特征融合
        for layer in self.ups:
            if isinstance(layer, Decoder_ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, segmap)

        # 上采样并进行特征融合
        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)

        if self.initial_pad != 0:
            return x[:, :, ip:-ip, ip:-ip]
        else:
            return x

# class UNet_2(nn.Module):    #x|0
#     def __init__(
#             self, img_channels, base_channels=16, channel_mults=(1,2,4 ,8),  # 输入图片太小 否则（1，2，4，8） base 128
#             num_res_blocks=2, time_emb_dim=16 * 4, time_emb_scale=1.0, num_classes=None, activation=SiLU(),
#             dropout=0.1, attention_resolutions=(1,),  num_groups=4, initial_pad=0,
#     ):
#         super().__init__()
#         # 使用到的激活函数，一般为SILU
#         self.activation = activation
#         # 是否对输入进行padding
#         self.initial_pad = initial_pad
#         # 需要去区分的类别数
#         self.num_classes = num_classes
#
#         # 对时间轴输入的全连接层
#         self.time_mlp = nn.Sequential(
#             PositionalEmbedding(base_channels, time_emb_scale),
#             nn.Linear(base_channels, time_emb_dim),
#             SiLU(),
#             nn.Linear(time_emb_dim, time_emb_dim),
#         ) if time_emb_dim is not None else None
#
#         # 对输入图片的第一个卷积
#         self.init_conv = nn.Conv2d(img_channels * 2, base_channels, 3, padding=1)
#
#         # self.downs用于存储下采样用到的层，首先利用ResidualBlock提取特征
#         # 然后利用Downsample降低特征图的高宽
#         self.downs = nn.ModuleList()  # 一系列下采样
#         self.ups = nn.ModuleList()  # 一系列上采样
#
#         # channels指的是每一个模块处理后的通道数
#         # now_channels是一个中间变量，代表中间的通道数
#         channels = [base_channels]
#         now_channels = base_channels
#         for i, mult in enumerate(channel_mults):
#             out_channels = base_channels * mult
#             for _ in range(num_res_blocks):
#                 self.downs.append(
#                     Encoder_ResidualBlock(
#                         now_channels, out_channels, dropout,
#                         time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
#                         num_groups=num_groups, use_attention=i in attention_resolutions,
#                     )
#                 )
#                 now_channels = out_channels
#                 channels.append(now_channels)
#
#             if i != len(channel_mults) - 1:
#                 self.downs.append(Downsample(now_channels))
#                 channels.append(now_channels)
#
#         # 可以看作是特征整合，中间的一个特征提取模块
#         self.mid = nn.ModuleList(
#             [
#                 Encoder_ResidualBlock(
#                     now_channels, now_channels, dropout,
#                     time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
#                      num_groups=num_groups, use_attention=True,
#                 ),
#                 Encoder_ResidualBlock(
#                     now_channels, now_channels, dropout,
#                     time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
#                      num_groups=num_groups, use_attention=False,
#                 ),
#             ]
#         )
#
#         # 进行上采样，进行特征融合
#         for i, mult in reversed(list(enumerate(channel_mults))):
#             out_channels = base_channels * mult
#
#             for _ in range(num_res_blocks + 1):
#                 self.ups.append(Encoder_ResidualBlock(
#                     channels.pop() + now_channels, out_channels, dropout,
#                     time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
#                      num_groups=num_groups, use_attention=i in attention_resolutions,
#                 ))
#                 now_channels = out_channels
#
#             if i != 0:
#                 self.ups.append(Upsample(now_channels))
#
#         assert len(channels) == 0
#
#         self.out_norm = nn.GroupNorm( num_groups, base_channels)
#         self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
#
#     def forward(self, x, time=None):
#         # 是否对输入进行padding
#         ip = self.initial_pad
#         if ip != 0:
#             x = F.pad(x, (ip,) * 4)
#
#         # 对时间轴输入的全连接层
#         if self.time_mlp is not None:
#             if time is None:
#                 raise ValueError("time conditioning was specified but tim is not passed")
#             time_emb = self.time_mlp(time)
#         else:
#             time_emb = None
#
#         # if self.num_classes is not None and y is None:
#         #     raise ValueError("class conditioning was specified but y is not passed")
#
#         # 对输入图片的第一个卷积
#
#         x = self.init_conv(x)
#
#         # skips用于存放下采样的中间层
#         skips = [x]
#         for layer in self.downs:
#             x = layer(x, time_emb)
#             skips.append(x)
#
#         # 特征整合与提取
#         for layer in self.mid:
#             x = layer(x, time_emb)
#
#         # 上采样并进行特征融合
#         for layer in self.ups:
#             if isinstance(layer, Encoder_ResidualBlock):
#                 x = torch.cat([x, skips.pop()], dim=1)
#             x = layer(x, time_emb)
#
#         # 上采样并进行特征融合
#         x = self.activation(self.out_norm(x))
#         x = self.out_conv(x)
#
#         if self.initial_pad != 0:
#             return x[:, :, ip:-ip, ip:-ip]
#         else:
#             return x

class UNet(nn.Module):
    def __init__(self, img_channels,):
        super().__init__()

        self.net_1 = UNet_1(img_channels)
        #self.net_2 = UNet_2(img_channels)
        self.hype = 4


    def forward(self, x,time, segmap):
        # print("x", x.shape)
        # print("time", time.shape)
        # print("segmap", segmap.shape)
        x_1 = self.net_1(x, time, segmap)
        x_2 = self.net_1(x, time)

        return x_1 + self.hype * (x_1 - x_2)










if __name__ == "__main__":
    save_path = r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\test\1.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PP = UNet(3).to(device)
    x = x0 = torch.randn(4, 6, 28, 28).to(device)
    segmap = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(1,10,(4,)).to(device)
    a = PP.forward(x,time = t,segmap=segmap)
    print("aa", a[0].shape)
    test_images = postprocess_output(a[0].cpu().data.numpy().transpose(1, 2, 0))
    print("aa", test_images.shape)
    #print("test_image", np.uint8(test_images)[:2])
    Image.fromarray(np.uint8(test_images)).save(save_path)