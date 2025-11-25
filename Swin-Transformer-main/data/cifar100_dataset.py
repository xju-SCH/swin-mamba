
import pickle
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CIFAR100Dataset(Dataset):
    """
    CIFAR-100 Dataset.
    """
    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train

        # 加载数据
        if train:
            file_path = os.path.join(root, 'train')
        else:
            file_path = os.path.join(root, 'test')

        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.images = data['data']
        self.labels = data['fine_labels']

        # CIFAR-100图像是32x32的，但需要重塑为(10000, 32, 32, 3)
        self.images = self.images.reshape(len(self.images), 3, 32, 32)
        self.images = self.images.transpose(0, 2, 3, 1)  # 转换为(10000, 32, 32, 3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]

        # 转换为PIL图像
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
