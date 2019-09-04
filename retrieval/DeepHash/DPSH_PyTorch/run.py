#!/usr/bin/env python
# -*- coding: utf-8 -*-

import DPSH
import data.dataloader as dataloader

import argparse
import torch
from loguru import logger


def run_dpsh(opt):
    """运行DLFH算法

    Parameters
        opt: parser
        程序运行参数

    Returns
        None
    """
    # 加载数据
    query_dataloader, train_dataloader, database_dataloader = dataloader.load_data(opt)

    logger.info(opt)

    # DLFH算法
    DPSH.dpsh(opt,
              train_dataloader,
              query_dataloader,
              database_dataloader,
              )


def load_parse():
    """加载程序参数

    Parameters
        None

    Returns
        opt: parser
        程序参数
    """
    parser = argparse.ArgumentParser(description='DPSH_PyTorch')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset used to train (default: cifar10)')
    parser.add_argument('--database_txt', default='./data/cifar10/database.txt', type=str,
                        help='path of cifar10 dataset')
    parser.add_argument('--train_txt', default='./data/cifar10/train.txt', type=str,
                        help='path of cifar10 dataset')
    parser.add_argument('--test_txt', default='./data/cifar10/test.txt', type=str,
                        help='path of cifar10 dataset')
    parser.add_argument('--num-query', default=10000, type=int,
                        help='number of query(default: 1000)')
    parser.add_argument('--num-train', default=50000, type=int,
                        help='number of train(default: 5000)')
    parser.add_argument('--code-length', default=48, type=int,
                        help='hyper-parameter: binary hash code length (default: 12)')
    parser.add_argument('--topk', default=5000, type=int,
                        help='compute map of top k (default: 5000)')
    parser.add_argument('--evaluate-freq', default=1, type=int,
                        help='frequency of evaluate (default: 10)')

    parser.add_argument('--model', default='alexnet', type=str,
                        help='CNN model(default: alexnet)')
    parser.add_argument('--multi-gpu', type=bool, default=False,
                        help='use multiple gpu')
    parser.add_argument('--gpu', default=0, type=int,
                        help='use gpu(default: 0. -1: use cpu)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate(default: 1e-5)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch size(default: 64)')
    parser.add_argument('--epochs', default=20, type=int,
                        help='epochs(default:64)')
    parser.add_argument('--num-workers', default=10, type=int,
                        help='number of workers(default: 4)')
    parser.add_argument('--eta', default=10, type=float,
                        help='hyper-parameter: regularization term (default: 10)')

    return parser.parse_args()


def set_seed(seed):
    import random
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


if __name__ == "__main__":
    opt = load_parse()
    logger.add('logs/file_{time}.log')

    # set_seed(20180707)

    if opt.gpu == -1:
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device("cuda:%d" % opt.gpu)

    run_dpsh(opt)
