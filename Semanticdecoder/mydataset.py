from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, training_images_path: list, segmap_images_path: list, ref_images_path: list, transform_three=None,transform_one = None):
        self.training_images_path = training_images_path
        self.segmap_images_path = segmap_images_path
        self.ref_images_path = ref_images_path
        self.transform_three = transform_three
        self.transform_one = transform_one

    def __len__(self):
        return len(self.training_images_path)

    def __getitem__(self, item):
        training_img = Image.open(self.training_images_path[item])
        segmap_img = Image.open(self.segmap_images_path[item])
        ref_img = Image.open(self.ref_images_path[item])
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # label = self.images_class[item]


        training_img = self.transform_three(training_img)


        segmap_img = self.transform_one(segmap_img)

        ref_img = self.transform_three(ref_img)


        return training_img, segmap_img, ref_img

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        training_images, segmap_images, ref_images = tuple(zip(*batch))

        training_images = torch.stack(training_images, dim=0)
        segmap_images = torch.stack(segmap_images, dim=0)
        ref_images = torch.stack(ref_images, dim=0)

        return training_images, segmap_images, ref_images

