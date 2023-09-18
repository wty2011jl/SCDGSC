from ddpmmodel import MyDDPM
import torch
from PIL import Image
import numpy as np
from utils import postprocess_output
from unetmodel import UNet
from torchvision import transforms
#from training import train_images_path, segmap_images_path, ref_images_path
class UnNormalize:
    #restore from T.Normalize
    #反归一化
    def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
        self.mean=torch.tensor(mean).view((1,-1,1,1))
        self.std=torch.tensor(std).view((1,-1,1,1))
    def __call__(self,x):
        x=(x*self.std)+self.mean
        return torch.clip(x,0,None)

data_transform = {
    "three": transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "one": transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(0.485, 0.229)])}

TT_1 = data_transform["three"]
TT_2 = data_transform["one"]
UTT_1 = UnNormalize(mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225))
UTT_2 = UnNormalize(mean= 0.485, std = 0.229)
def generate_1x1_image( ddpm, save_path, segmap, reference):
    with torch.no_grad():
        randn_in = torch.randn((1, 1)).to(device)
        test_images_ = ddpm.sample(1, randn_in.device,segmap = segmap, reference = reference, use_ema=False)
        #print("textimage", test_images_.shape)
        #print("oringal", test_images_[0], sum(sum(sum(test_images_[0]))))
        test_images = UTT_1(test_images_.cpu()).data.numpy()
        #print("UTT1", test_images[0], sum(sum(sum(test_images[0]))))
        #print("cdc", sum(sum(sum(test_images - test_images_))))
        test_images = test_images * 255
        test_images = np.squeeze(test_images,axis=0).transpose(1,2,0)
        #test_images.save(save_path)
        #test_images = postprocess_output(test_images[0].cpu().data.numpy())#.transpose(2,0,1)
        test_images = np.uint8(test_images)
        #print("uint8", test_images,sum(sum(sum(test_images))))
        #print("test_image",np.uint8(test_images).shape)
        Image.fromarray(test_images,"RGB").save(save_path)  #通道顺序自动变化

if __name__ == "__main__":
    save_path_5x5 = "results/predict_out/predict_5x5_results.png"
    save_path_1x1 = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results\predict_1x1_results.png"

    model_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\haze\ddpm_model_one_reference_s_4_decoder_200-400epoch.pt"
    image_chw = (3,96,128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MyUNet = UNet(3).to(device)
    ddpm = MyDDPM(MyUNet, n_steps=1000, min_beta=10 ** (-4), max_bata=0.02, device=device, image_chw=image_chw)  # !!!
    weights_dict = torch.load(model_path, map_location='cpu')
    ddpm.load_state_dict(weights_dict)
    print("d")
    ddpm.to(device)
    segmap_ = Image.open(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_second_reference\segmap_image\1_1057.png")
    segmap = TT_2(segmap_)
    segmap = segmap.unsqueeze(0)
    #print("segmap", segmap.shape)
    #segmap = np.array(segmap_)[None,None,:,:,]
    print("segmap", segmap.shape)
    reference_ = Image.open(r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\dataset_forth_reference\reference_image\4_71.jpg")
    reference = TT_1(reference_)
    reference = reference.unsqueeze(0)
    #reference = np.array(reference_)[None,:,:,:]
    #reference = reference.transpose(0,3,1,2)
    print("gggg", reference.shape)
    segmap = torch.FloatTensor(segmap).to(device)
    reference = torch.FloatTensor(reference).to(device)
    # while True:
    #     img = input('Just Click Enter~')
    #     print("Generate_1x1_image")
    #     generate_1x1_image(ddpm, save_path_1x1, segmap, reference)
    #
    #     print("Generate_1x1_image Done")
    for i in range(100):
        with torch.no_grad():
            randn_in = torch.randn((1, 1)).to(device)
            ta,tb,tc,td,te,tf,tg,th, ti, tj = ddpm.sample(1, randn_in.device, segmap=segmap, reference=reference, use_ema=False)
            ta = UTT_1(ta.cpu()).data.numpy()
            ta = ta * 255
            ta = np.squeeze(ta, axis=0).transpose(1, 2, 0)
            ta = np.uint8(ta)
            #print("dd", ta)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dA.jpg" %i
            Image.fromarray(ta, "RGB").save(save_path)  # 通道顺序自动变化

            tb = UTT_1(tb.cpu()).data.numpy()
            tb = tb * 255
            tb = np.squeeze(tb, axis=0).transpose(1, 2, 0)
            tb = np.uint8(tb)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dB.jpg" % i
            Image.fromarray(tb, "RGB").save(save_path)  # 通道顺序自动变化

            tc = UTT_1(tc.cpu()).data.numpy()
            tc = tc * 255
            tc = np.squeeze(tc, axis=0).transpose(1, 2, 0)
            tc = np.uint8(tc)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dC.jpg" % i
            Image.fromarray(tc, "RGB").save(save_path)  # 通道顺序自动变化

            td = UTT_1(td.cpu()).data.numpy()
            td = td * 255
            td = np.squeeze(td, axis=0).transpose(1, 2, 0)
            td = np.uint8(td)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dD.jpg" % i
            Image.fromarray(td, "RGB").save(save_path)  # 通道顺序自动变化

            te = UTT_1(te.cpu()).data.numpy()
            te = te * 255
            te = np.squeeze(te, axis=0).transpose(1, 2, 0)
            te = np.uint8(te)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dE.jpg" % i
            Image.fromarray(te, "RGB").save(save_path)  # 通道顺序自动变化

            tf = UTT_1(tf.cpu()).data.numpy()
            tf = tf * 255
            tf = np.squeeze(tf, axis=0).transpose(1, 2, 0)
            tf = np.uint8(tf)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dF.jpg" % i
            Image.fromarray(tf, "RGB").save(save_path)  # 通道顺序自动变化

            tg = UTT_1(tg.cpu()).data.numpy()
            tg = tg * 255
            tg = np.squeeze(tg, axis=0).transpose(1, 2, 0)
            tg = np.uint8(tg)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dG.jpg" % i
            Image.fromarray(tg, "RGB").save(save_path)  # 通道顺序自动变化

            th = UTT_1(th.cpu()).data.numpy()
            th = th * 255
            th = np.squeeze(th, axis=0).transpose(1, 2, 0)
            th = np.uint8(th)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dH.jpg" % i
            Image.fromarray(th, "RGB").save(save_path)  # 通道顺序自动变化
            ti = UTT_1(ti.cpu()).data.numpy()
            ti = ti * 255
            ti = np.squeeze(ti, axis=0).transpose(1, 2, 0)
            ti = np.uint8(ti)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dI.jpg" % i
            Image.fromarray(ti, "RGB").save(save_path)  # 通道顺序自动变化
            tj = UTT_1(tj.cpu()).data.numpy()
            tj = tj * 255
            tj = np.squeeze(tj, axis=0).transpose(1, 2, 0)
            tj = np.uint8(tj)
            save_path = r"C:\Users\user\Desktop\change_driven SemCom\Semanticdecoder\results220\s%dJ.jpg" % i
            Image.fromarray(tj, "RGB").save(save_path)  # 通道顺序自动变化
        print("image generated",i)
