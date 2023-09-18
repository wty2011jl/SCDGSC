import os

import torch
from torchvision import transforms

from mydataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
root = r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset1"  # 数据集所在根目录


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    training_images_path, segmap_images_path = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    # nn = len(train_images_path)
    # print(nn)
    batch_size = 16
    train_data_set = MyDataSet(training_images_path = training_images_path,
                               segmap_images_path = segmap_images_path,
                               transform=data_transform["train"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    #plot_data_loader_image(train_loader)

    # for step, data in enumerate(train_loader):
    #     images = data


if __name__ == '__main__':
    main()
