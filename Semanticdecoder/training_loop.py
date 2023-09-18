import numpy as np

from ddpmmodel import MyDDPM
from unetmodel import UNet
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import datetime

def training_loop(
        ddpm,
        loader,
        n_epochs,
        optim,
        device,
        display = False,
        eta = None,
        store_path = "ddpm_model_one_reference_s_4_decoder_200-400epoch.pt"
):
    #mse = nn.MSELoss()
    best_loss = float("inf")  #新的best loss比这个小
    #n_steps = ddpm.n_steps
    loss_history_s_decoder_4_200_400 = np.zeros(n_epochs)
    results_file = "decoder_resultss4 200_400{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    for epoch in tqdm(range(n_epochs), desc=f"Training process", colour="#00ff00"):
        epoch_loss = 0.0

        #for step, batch in enumerate(tqdm(loader,leave=False,desc=f"Eopch {epoch + 1}/{n_epochs}", colour = "#00ff00" )):
        for step, batch in enumerate(loader):
            #torch.cuda.empty_cache()
            x0, segmap, ref = batch
            #print("ddd", segmap.size())
            x0 = x0.to(device)  #图像原图

            segmap = segmap.to(device)

            ref = ref.to(device)



            # x0 = torch.randn(4, 3, 28, 28).to(device)  #adaptive to the fake  batch_size=4
            # segmap = torch.randn(4,1,28,28).to(device)
            #reference = torch.randn(4, 3, 28, 28).to(device)
            #n = len(x0)   #batchsize

            # #forward 输入： x0 （原图）， eta, 噪声, t
            # noisy_imgs = ddpm(x0, t, eta)  #x0 eta: 128, 1, 28, 28
            #
            # eta_theta = ddpm.backward(noisy_imgs, t) # the predicted noise

            optim.zero_grad()
            loss = torch.mean(ddpm.forward(x0,segmap,ref))# 默认forward函数

            loss.backward()
            optim.step()
            # loss = MyDDPM.forward(x0)
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            ddpm.update_ema()
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)   #dataset总长度
        #Display images generated at this epoch
        # if display:
        #     show_image(generate_new_images(ddpm, device=device),f"Images generated at epoch {epoch} + 1")
        loss_history_s_decoder_4_200_400[epoch] = epoch_loss
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {epoch_loss:.4f}\n" \

            s =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            f.write(train_info + s + "\n\n")
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        #save the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += "--> Best model ever(stored)"  #链接字符串

        print(log_string)
    end_store_path = "end_ddpm_model_one_reference_s_4_200_400epoch.pt"
    torch.save(ddpm.state_dict(), end_store_path)
    np.save("loss_history_s_decoder_4_0_200", loss_history_s_decoder_4_200_400)