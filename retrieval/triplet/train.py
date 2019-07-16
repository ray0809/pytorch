
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
from tqdm import tqdm
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from loss import TripleLoss
from data_loader import TripletDataset
from model import load_resnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torchvision.set_image_backend('accimage')



def main():


    print('>>>开始制作生成器')
    traingen = TripletDataset('./txt/train.txt', tag='train', transforms=transforms.Compose([
                                                                        transforms.Resize(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.RandomHorizontalFlip(0.5),
                                                                        transforms.ToTensor()]))
    trainloader = DataLoader(dataset=traingen,
                            batch_size=96,
                            pin_memory=False,
                            drop_last=False,
                            shuffle=True,
                            num_workers=10)
    testgen = TripletDataset('./txt/test.txt', tag='test', transforms=transforms.Compose([transforms.Resize((224,224)),
                                                                                        transforms.ToTensor()]))
    testloader = DataLoader(dataset=testgen,
                            batch_size=96,
                            pin_memory=False,
                            drop_last=False,
                            shuffle=True,
                            num_workers=10)


    print('>>>开始测试生成器')
    temp_train = iter(trainloader)
    temp_test = iter(testloader)
    img1, img2, img3 = next(temp_train)
    print('    >>>train img1:{}, img2:{}, img3:{}'.format(img1.shape, img2.shape, img3.shape))
    img1, img2, img3 = next(temp_test)
    print('    >>>test img1:{}, img2:{}, img3:{}'.format(img1.shape, img2.shape, img3.shape))
    del temp_test, temp_train


    print('>>>开始加载模型')
    model = load_resnet(num_classes=128)
    model = model.to(device)
    criterion = TripleLoss(margin=1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)


    print('>>>开始训练')
    epochs = 20
    INF =  np.inf
    for i in range(epochs):
        total_loss = 0
        train_count = 0
        for (img1, img2, img3) in tqdm(trainloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            # print(tag)
            a = model(img1)
            p = model(img2)
            n = model(img3)
            loss = criterion(a, p, n)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count +=1 
        
        
        
        with torch.no_grad():
            model.eval()
            test_total_loss = 0
            test_count = 0
            for (img1, img2, img3) in tqdm(testloader):
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)
                # print(tag)
                a = model(img1)
                p = model(img2)
                n = model(img3)
                loss = criterion(a, p, n)
                test_total_loss += loss.item()
                test_count += 1
            val_loss = test_total_loss / test_count
        
        print('{}/{}, train_loss:{}'.format(i+1, epochs, total_loss / train_count))
        print('{}/{}, val_loss:{}'.format(i+1, epochs, val_loss))

        if loss < INF:
            INF = loss
            if os.path.isfile('./weight/resnet34_cifar10.ckpt'):
                os.remove('./weight/resnet34_cifar10.ckpt')
            torch.save(model.state_dict(),'./weight/resnet34_cifar10.ckpt')



if __name__ == '__main__':
    main()