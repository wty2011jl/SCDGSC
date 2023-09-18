from ddpmmodel import MyDDPM
import torch
from PIL import Image
import numpy as np
from utils import postprocess_output
from unetmodel import UNet

def generate_1x1_image( ddpm, save_path, segmap, reference):
    with torch.no_grad():
        randn_in = torch.randn((1, 1)).to(device)
        test_images = ddpm.sample(1, randn_in.device,segmap = segmap, reference = reference, use_ema=False)
        test_images = postprocess_output(test_images[0].cpu().data.numpy()).transpose(2, 1, 0)
        print("test_image",np.uint8(test_images).shape)
        Image.fromarray(np.uint8(test_images[:,:,:])).save(save_path)  #通道顺序自动变化

if __name__ == "__main__":
    save_path_5x5 = "results/predict_out/predict_5x5_results.png"
    save_path_1x1 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results\predict_1x1_results.png"

    model_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\ddpm_model.pt"
    image_chw = (3,128,96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MyUNet = UNet(3).to(device)
    ddpm = MyDDPM(MyUNet, n_steps=1000, min_beta=10 ** (-4), max_bata=0.02, device=device, image_chw=image_chw)  # !!!
    weights_dict = torch.load(model_path, map_location='cpu')
    ddpm.load_state_dict(weights_dict)
    print("d")
    ddpm.to(device)
    segmap_ = Image.open(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\segmap_image\1_71.png")
    #print("segmap", segmap_)
    segmap = np.array(segmap_)[None,None,:,:].transpose(0,1,3,2)
    print("segmap", segmap.shape)
    reference_ = Image.open(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_min\reference_image\1_71.jpg")
    reference = np.array(reference_)[None,:,:,:]
    print("gggg", reference.shape)
    reference = reference.transpose(0,3,2,1)

    segmap = torch.FloatTensor(segmap).to(device)
    reference = torch.FloatTensor(reference).to(device)
    while True:
        img = input('Just Click Enter~')
        print("Generate_1x1_image")
        generate_1x1_image(ddpm, save_path_1x1, segmap, reference)

        print("Generate_1x1_image Done")