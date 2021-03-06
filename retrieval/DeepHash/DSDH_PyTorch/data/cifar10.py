# -*- coding: utf-8 -*-

from data.transform import *
# from data.transform import Onehot

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import DataLoader


import os
import sys
import pickle


def load_data_gist(path, train=True):
    """加载对cifar10使用gist提取的数据

    Parameters
        path: str
        数据路径

        train: bools
        True，加载训练数据; False，加载测试数据

    Returns
        data: ndarray
        数据

        labels: ndarray
        标签
    """
    mat_data = sio.loadmat(path)

    if train:
        data = mat_data['traindata']
        labels = mat_data['traingnd'].astype(np.int)
    else:
        data = mat_data['testdata']
        labels = mat_data['testgnd'].astype(np.int)

    return data, labels


# def load_data(opt):
#     """加载cifar10数据

#     Parameters
#         opt: Parser
#         配置

#     Returns
#         query_dataloader, train_dataloader, database_dataloader: DataLoader
#         数据加载器
#     """
#     CIFAR10.init(opt.data_path, opt.num_query, opt.num_train)
#     query_dataset = CIFAR10('query', transform=img_transform(), target_transform=Onehot())
#     train_dataset = CIFAR10('train', transform=img_transform(), target_transform=Onehot())
#     database_dataset = CIFAR10('database', transform=img_transform(), target_transform=Onehot())

#     query_dataloader = DataLoader(query_dataset,
#                                   batch_size=opt.batch_size,
#                                   num_workers=opt.num_workers,
#                                   )
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=opt.batch_size,
#                                   num_workers=opt.num_workers,
#                                   )
#     database_dataloader = DataLoader(database_dataset,
#                                      batch_size=opt.batch_size,
#                                      num_workers=opt.num_workers,
#                                      )

#     return query_dataloader, train_dataloader, database_dataloader


# class CIFAR10(data.Dataset):
#     """加载官网下载的CIFAR10数据集"""
#     @staticmethod
#     def init(root, num_query, num_train):
#         data_list = ['data_batch_1',
#                      'data_batch_2',
#                      'data_batch_3',
#                      'data_batch_4',
#                      'data_batch_5',
#                      'test_batch',
#                      ]
#         base_folder = 'cifar-10-batches-py'

#         data = []
#         targets = []

#         for file_name in data_list:
#             file_path = os.path.join(root, base_folder, file_name)
#             with open(file_path, 'rb') as f:
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(f)
#                 else:
#                     entry = pickle.load(f, encoding='latin1')
#                 data.append(entry['data'])
#                 if 'labels' in entry:
#                     targets.extend(entry['labels'])
#                 else:
#                     targets.extend(entry['fine_labels'])

#         data = np.vstack(data).reshape(-1, 3, 32, 32)
#         data = data.transpose((0, 2, 3, 1))  # convert to HWC
#         targets = np.array(targets)

#         CIFAR10.ALL_IMG = data
#         CIFAR10.ALL_TARGETS = targets

#         # split data, tags
#         perm_index = np.random.permutation(CIFAR10.ALL_IMG.shape[0])
#         query_index = perm_index[:num_query]
#         train_index = perm_index[:num_train]

#         CIFAR10.QUERY_IMG = CIFAR10.ALL_IMG[query_index, :]
#         CIFAR10.QUERY_TARGETS = CIFAR10.ALL_TARGETS[query_index]
#         CIFAR10.TRAIN_IMG = CIFAR10.ALL_IMG[train_index, :]
#         CIFAR10.TRAIN_TARGETS = CIFAR10.ALL_TARGETS[train_index]

#     def __init__(self, mode='train',
#                  transform=None, target_transform=None,
#                  ):
#         self.transform = transform
#         self.target_transform = target_transform

#         if mode == 'train':
#             self.img = CIFAR10.TRAIN_IMG
#             self.targets = CIFAR10.TRAIN_TARGETS
#         elif mode == 'query':
#             self.img = CIFAR10.QUERY_IMG
#             self.targets = CIFAR10.QUERY_TARGETS
#         else:
#             self.img = CIFAR10.ALL_IMG
#             self.targets = CIFAR10.ALL_TARGETS

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target, index) where target is index of the target class.
#         """
#         img, target = self.img[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target, index

#     def __len__(self):
#         return len(self.img)




def load_data(opt):
    """加载cifar10数据

    Parameters
        opt: Parser
        配置

    Returns
        query_dataloader, train_dataloader, database_dataloader: DataLoader
        数据加载器
    """
    query_dataset = CIFAR10(opt.test_txt, transform=test_img_transform(), target_transform=Onehot())
    train_dataset = CIFAR10(opt.train_txt, transform=train_img_transform(), target_transform=Onehot())
    database_dataset = CIFAR10(opt.database_txt, transform=test_img_transform(), target_transform=Onehot())

    query_dataloader = DataLoader(query_dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  )
    database_dataloader = DataLoader(database_dataset,
                                     batch_size=opt.batch_size,
                                     num_workers=opt.num_workers,
                                     )

    return query_dataloader, train_dataloader, database_dataloader


class CIFAR10(data.Dataset):
    
    

    def __init__(self, txt,
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform
        self.img = []
        self.targets = []
        with open(txt, 'r') as f:
            for i in f.readlines():
                i = i.strip().split()
                self.img.append(i[0])
                self.targets.append(int(i[1]))

        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.img[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.img)