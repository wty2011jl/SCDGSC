import os
import time
import json
import cv2

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import lraspp_mobilenetv3_large


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(N):
    classes = 1
    weights_path = r"C:\Users\12955\Desktop\change_driven SemCom\lraspp\save_weights\model_9.pth"
    img_path = r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\JPEGImages\8_%d.jpg" % N
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = lraspp_mobilenetv3_large(num_classes=classes+1)

    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        #mask.save(r"C:\Users\12955\Desktop\change_driven SemCom\Video_Generation\results1\1_%d.png" % N)
        mask.save(r"C:\Users\12955\Desktop\change_driven SemCom\Video_Generation\8_%d.png" % N)

if __name__ == '__main__':
    main(200)
#
# a = cv2.imread(r"C:\Users\12955\Desktop\change_driven SemCom\VOC_CGSC\JPEGImages\4_1200.jpg")
# c = cv2.resize(a,(693,520))
# cv2.imwrite(r"C:\Users\12955\Desktop\change_driven SemCom\lraspp\Simulation\4_pred.png",c)