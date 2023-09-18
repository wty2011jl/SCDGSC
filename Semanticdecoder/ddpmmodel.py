import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import *




#UNet model

# class FakeUNet(nn.Module):
#     def __init__(self):
#         super (FakeUNet, self).__init__()
#         self.linear = nn.Linear(28 * 28, 28 * 28)
#         self.relu = nn.ReLU()
#
#     def forward (self, x, t, y):
#         x = x.view(x.size(0),-1)  #第0维度的值   Flatten the input
#         x = self.linear(x)
#         x = self.relu(x)
#         return x.view(x.size(0), 1, 28, 28)  #Reshape back to the original shape

#DDPM model
class EMA():
    def __init__(self, decay):
        self.decay = decay

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_network, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_network.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)


class MyDDPM(nn.Module):
    def __init__(self, network, n_steps = 1000, min_beta = 10 ** (-4), max_bata = 0.02, device = None, image_chw=None,ema_decay = 0.9, ema_start = 10, ema_update_rate = 10):
        super (MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.step = 0
        self.ema_decay = ema_decay
        self.ema = EMA(ema_decay)
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.ema_network = deepcopy(network)
        # betas = torch.linspace(min_beta, max_bata, n_steps)
        # self.betas = torch.zeros(n_steps + 1)   #考虑到原始图像本身
        # self.betas[1:] = betas
        # self.betas[0] = 10**(-15)
        # self.betas = self.betas.to(device)
        self.betas = torch.linspace(min_beta, max_bata, n_steps).to(device)
        self.loss_type = "l2"
        self.alphas = (1 - self.betas).to(self.device)
        self.alphas_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range (len(self.alphas))]).to(device)
        self.sub_alpha_bars = (1 - self.alphas_bars).to(self.device)
        self.remove_noise_coeff = (self.betas / torch.sqrt(1 - self.alphas_bars)).to(self.device)
        self.reciprocal_sqrt_alphas = torch.sqrt_(1 / self.alphas)
        self.sigma = torch.sqrt(self.betas).to(self.device)
        self.lamda = 1.1
#training
    def perturb_x(self, x, t, noise):   #x0原图 目标图
        # print("dd", self.device)
        # print("ss", self.alphas_bars.device)
        # print("dd", self.sub_alpha_bars.device)
        return (
                extract(self.alphas_bars, t, x.shape) * x +
                extract(self.sub_alpha_bars, t, x.shape) * noise
        )

    @torch.no_grad()
    def remove_noise_1(self, x, t, segmep, reference, use_ema=True):
        # print("ssss", x.size(), segmep.size(), reference.size())
        x_hat = torch.cat((x, reference), dim=1)
        if use_ema:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_network(x_hat, t, segmep)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                    (x - extract(self.remove_noise_coeff, t, x.shape) * self.network(x_hat, t, segmep)) *
                    extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )


    # def remove_noise(self, x, t, segmep, reference, use_ema=True):
    #     # print("ssss", x.size(), segmep.size(), reference.size())
    #     x_hat = torch.cat((x, reference), dim=1)
    #     if use_ema:
    #         return (
    #                 (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_network(x_hat, t, segmep)) *
    #                 extract(self.reciprocal_sqrt_alphas, t, x.shape)
    #         )
    #     else:
    #         return (
    #                 (x - extract(self.remove_noise_coeff, t, x.shape) * self.network(x_hat, t, segmep)) *
    #                 extract(self.reciprocal_sqrt_alphas, t, x.shape)
    #         )

    # def get_losses(self, x, t, segmap, reference):
    #     # x, noise [batch_size, 3, 64, 64]
    #     noise_1 = torch.randn_like(x)
    #     noise_2 = torch.randn_like(x)
    #     t_ = t - 1
    #     perturbed_x_1 = self.perturb_x(x, t_, noise_1)   #noisy x_{t-1} (q(x_{t-1}|x_{t}, x_0))
    #     perturbed_x  = extract(self.alphas, t, perturbed_x_1.shape) * perturbed_x_1 + extract(self.betas, t, perturbed_x_1.shape) * noise_2 #q(x_{t})
    #
    #     x_= torch.cat((perturbed_x,reference), dim = 1)
    #
    #     estimated_noise = self.network(x_, t, segmap)
    #     estimated_x_1 = self.remove_noise(perturbed_x, t, segmap,reference, use_ema=False)
    #     estimated_x_1 += extract(self.sigma, t, x.shape) * torch.randn_like(estimated_x_1)
    #     if self.loss_type == "l1":
    #         loss_1 = F.l1_loss(estimated_noise, noise_2)
    #         loss_2 = F.l1_loss(estimated_x_1, perturbed_x_1)
    #     elif self.loss_type == "l2":
    #         loss_1 = F.mse_loss(estimated_noise, noise_2)
    #         loss_2 = F.mse_loss(estimated_x_1, perturbed_x_1)
    #     return loss_1 + self.lamda * loss_2

    def get_losses(self, x, t, segmap, reference):
        # x, noise [batch_size, 3, 64, 64]
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)

        x_= torch.cat((perturbed_x,reference), dim = 1)

        estimated_noise = self.network(x_, t, segmap)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)
        return loss

    # def get_losses(self, x, t, segmap, reference):
    #     # x, noise [batch_size, 3, 64, 64]
    #     noise_1 = torch.randn_like(x)
    #     noise_2 = torch.randn_like(x)
    #     t_ = t - 1
    #     perturbed_x_1 = self.perturb_x(x, t_, noise_1)   #noisy x_{t-1} (q(x_{t-1}|x_{t}, x_0))
    #     perturbed_x  = extract(self.alphas, t, perturbed_x_1.shape) * perturbed_x_1 + extract(self.betas, t, perturbed_x_1.shape) * noise_2 #q(x_{t})
    #
    #     #x_= torch.cat((perturbed_x,reference), dim = 1)
    #
    #     #estimated_noise = self.network(x_, t, segmap)
    #     estimated_x_1 = self.remove_noise(perturbed_x, t, segmap,reference, use_ema=False)
    #     estimated_x_1 += extract(self.sigma, t, x.shape) * torch.randn_like(estimated_x_1)
    #     if self.loss_type == "l1":
    #         #loss_1 = F.l1_loss(estimated_noise, noise_2)
    #         loss_2 = F.l1_loss(estimated_x_1, perturbed_x_1)
    #     elif self.loss_type == "l2":
    #         #loss_1 = F.mse_loss(estimated_noise, noise_2)
    #         loss_2 = F.mse_loss(estimated_x_1, perturbed_x_1)
    #     return  loss_2

    def forward(self, x, segmap=None, reference = None):
        b, c, h, w = x.shape
        device = x.device

        # if h != self.img_size[0]:
        #     raise ValueError("image height does not match diffusion parameters")
        # if w != self.img_size[0]:
        #     raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, self.n_steps, (b,), device=device)    # 1, n_steps + 1  /  0, n_step
        return self.get_losses(x, t, segmap, reference)
#*参数更新
    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_network.load_state_dict(self.network.state_dict())
            else:
                self.ema.update_model_average(self.ema_network, self.network)

#sampling


    @torch.no_grad()
    def sample(self, batch_size, device, segmap=None, reference = None, use_ema=True):
        # if segmap is not None and batch_size != len(segmap):
        #     raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size,  *self.image_chw, device=device)  # img_size:

        #print("xx", x)   #高斯白噪声
        for t in range(self.n_steps -1 , -1 , -1):  #reversed(range(...))   #前面改成n_step + 1   n_setp  -1
            #print("step",t)
            t_batch = torch.tensor([t], device=device).repeat(batch_size)  #没有[]也可以
            x = self.remove_noise_1(x, t_batch, segmap, reference, use_ema)  #%
            if t == 999:
                x_999 = x
            if t == 574:
                x_574 = x
            if t == 432:
                x_432 = x
            if t == 395:
                x_395 = x
            if t == 360:
                x_360 = x
            if t == 320:
                x_320 = x
            if t == 290:
                x_290 = x
            if t == 148:
                x_148 = x
            if t == 70:
                x_70 = x
            if t > 1:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            #print("sum",sum(sum(x)))
        return x_999.cpu().detach(), x_574.cpu().detach(), x_432.cpu().detach(), x_395.cpu().detach(), x_360.cpu().detach(), x_320.cpu().detach(), x_290.cpu().detach(), x_148.cpu().detach(), x_70.cpu().detach(), x.cpu().detach()
    # def forward(self, x0, t,eta = None): #batchsize = 128  x0 128, 1 , 28 * 28  (神经网络的输入)
    #
    #     n, c, h, w = x0.shape
    #     a_bar = self.alphas_bars[t]
    #
    #     if eta is None:
    #         eta  = torch.randn(n,c,h,w).to(self.device)
    #
    #     noisy = a_bar.sqrt().reshape(n,1,1,1) * x0 + (1 - a_bar).sqrt().reshape(n,1,1,1) * eta
    #
    #     return noisy  #noisy image

    # def backward(self, x, t):
    #     return self.network(x, t)
    # pass




if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fake_unet = FakeUNet().to(device)
    # ddpm = MyDDPM(fake_unet, device=device)
    #
    # x0 = torch.randn(1,1,28,28).to(device)
    # t = torch.randint(0, ddpm.n_steps,(1,)).to(device)
    #
    # noisy_image = ddpm.forward(x0,t)
    # print("Forward: noisy_image.shape:", noisy_image.shape)
    #
    # predict_noise = ddpm.backward(noisy_image,t)
    # print("Backward: predict_noise.shape", predict_noise.shape)
    for t in range(7, -1, -1):
        print("t", t)


