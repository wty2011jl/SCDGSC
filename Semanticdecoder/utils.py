import os
import json
import pickle
import random

import matplotlib.pyplot as plt

def extract(a, t, x_shape):
    #print("aaa",a.device, t.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # b,1,1

#dataset
def read_split_data(root_training: str, root_segmap: str, root_ref: str, val_rate: float = 0):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root_training), "dataset root: {} does not exist.".format(root_training)
    assert os.path.exists(root_segmap), "dataset root: {} does not exist.".format(root_segmap)
    assert os.path.exists(root_ref), "dataset root: {} does not exist.".format(root_ref)

    # 遍历文件夹，一个文件夹对应一个类别
    # flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # # 排序，保证顺序一致
    # flower_class.sort()
    # # 生成类别名称以及对应的数字索引
    # class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    training_images_path = []  # 存储训练集的所有图片路径
    segmap_images_path = [] #
    ref_images_path = []
    # train_images_label = []  # 存储训练集图片对应索引信息
    # val_images_path = []  # 存储验证集的所有图片路径
    # val_images_label = []  # 存储验证集图片对应索引信息
    # every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    # for cla in flower_class:
    #     cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
    training_images = [os.path.join(root_training, i) for i in os.listdir(os.path.join(root_training))
                  if os.path.splitext(i)[-1] in supported]
    training_images.sort()
    segmap_images = [os.path.join(root_segmap, i) for i in os.listdir(os.path.join(root_segmap))
                        if os.path.splitext(i)[-1] in supported]
    segmap_images.sort()
    ref_images = [os.path.join(root_ref, i) for i in os.listdir(os.path.join(root_ref))
                        if os.path.splitext(i)[-1] in supported]
    ref_images.sort()
        # 获取该类别对应的索引
        #image_class = class_indices[cla]
        # 记录该类别的样本数量
        #every_class_num.append(len(images))
        # 按比例随机采样验证样本
        #val_path = random.sample(images, k=int(len(images) * val_rate))

    for img_path in training_images:
            # if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            #     val_images_path.append(img_path)
            #     val_images_label.append(image_class)
            # else:  # 否则存入训练集
        training_images_path.append(img_path)
                #train_images_label.append(image_class)
    for img_path in segmap_images:
        segmap_images_path.append(img_path)
    for img_path in ref_images:
        ref_images_path.append(img_path)


    #print("{} images were found in the dataset.".format(sum(every_class_num)))
    #print("{} images for segmap.".format(len(segmap_images_path)))
    #print("{} images for validation.".format(len(val_images_path)))

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()


    return training_images_path, segmap_images_path, ref_images_path


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), json_path + " does not exist."
    # json_file = open(json_path, 'r')
    # class_indices = json.load(json_file)

    for data in data_loader:
        images = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            #label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel("class_indices[str(label)]")
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def preprocess_input(x):
    x /= 255
    x -= 0.5
    x /= 0.5
    return x

def postprocess_output(x):
    x *= 0.5
    x += 0.5
    x *= 255
    return x


if __name__ == "__main__":
    X = read_split_data(root= r"C:\Users\12955\Desktop\change_driven SemCom\Semanticdecoder\dataset1")
    print("X", X)



