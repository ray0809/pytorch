import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data_loader import TripletDataset, BalancedBatchSampler
from model import MarginNet
from loss import Margin_Loss
from util import cal_map



parser = argparse.ArgumentParser(description="train a margin based loss model")
parser.add_argument('--train_txt',type=str,default="./txt/train.txt",
                    help='train path of the cifar10')
parser.add_argument('--test_txt',type=str,default="./txt/test.txt",
                    help='test path of the cifar10')
parser.add_argument('--embed_dim',type=int,default=128,
                    help='dimensionality of image embeding,times of 8')
parser.add_argument('--batch_size',type=int,default=200,
                    help='training batch size per device')
parser.add_argument('--batch_k',type=int,default=20,
                    help='number of images per class in a batch,can be divided by batch_size')
parser.add_argument('--gpu_ids',type=str,default='0,1',
                    help='the gpu_id of the runing batch')
parser.add_argument('--epochs',type=int,default=40,
                    help='number of training epochs,default is 100')
parser.add_argument('--lr',type=float,default=0.005,
                    help='learning rate of the resnet and dense layer')
parser.add_argument('--lr_beta',type=float,default=0.1,
                    help='learning rate for the beta in margin based loss')
parser.add_argument('--margin',type=float,default=0.2,
                    help='margin for the margin based loss,default is 0.2')
parser.add_argument('--beta',type=float,default=1.2,
                    help='the class specific beta parameter')
parser.add_argument('--nu',type=float,default=0.001,
                    help='regularization parameter for beta,default is 0')
parser.add_argument('--steps',type=str,default='10,20,30,40',
                    help='epochs to updata learning rate')
parser.add_argument('--seed',type=int,default=123,
                    help='random seed to use,default=123')
parser.add_argument('--factor',type=float,default=0.5,
                    help='learning rate schedule factor,default is 0.5')


opt = parser.parse_args()
torch.random.manual_seed(opt.seed)
np.random.seed(opt.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
steps = [int(i) for i in opt.steps.split(',')]
gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
if len(gpu_ids) == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])

def train():


    print('>>>加载dataloader')
    traingen = TripletDataset(opt.train_txt, transforms=transforms.Compose([transforms.Resize((224, 224)),
                                                                            transforms.RandomRotation(10),
                                                                            transforms.RandomHorizontalFlip(0.5),
                                                                            transforms.ToTensor()]))
    trainloader = DataLoader(dataset=traingen,
                            batch_sampler=BalancedBatchSampler(opt.train_txt, opt.batch_size, opt.batch_k),
                            num_workers=10,
                            pin_memory=False)


    testgen = TripletDataset(opt.test_txt, transforms=transforms.Compose([transforms.Resize((224,224)),
                                                            transforms.ToTensor()]))
    testloader = DataLoader(dataset=testgen,
                            batch_size=opt.batch_size,
                            num_workers=10,
                            pin_memory=False)



    # model = load_resnet(num_classes=opt.embed_dim)
    model = MarginNet(embedding_dim=opt.embed_dim)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    beta = torch.tensor(np.ones(opt.batch_size)*opt.beta, requires_grad=True, dtype=torch.float32, device=device)

    criterion = Margin_Loss(batch_k=opt.batch_k, margin=opt.margin, nu=opt.nu)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opt.lr, momentum=0.9)
    optimizer_beta = torch.optim.SGD([{'params':beta}], lr=opt.lr_beta, momentum= 0.9)

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=opt.factor)


    print('>>>开始训练')
    epochs = opt.epochs
    INF =  0
    for i in range(epochs):
        total_loss = 0
        train_count = 0
        model.train()
        for imgs, targets in tqdm(trainloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            embedding = model(imgs)

            loss = criterion(embedding, targets, beta)
            total_loss += loss.item()
            optimizer.zero_grad()
            optimizer_beta.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_beta.step()
            train_count +=1

        



        with torch.no_grad():
            model.eval()
            feats = []
            targets = []
            for (imgs, target) in testloader:
                imgs = imgs.to(device)
                target = target.numpy().tolist()
                feat = model(imgs).detach().cpu().numpy().tolist()
                feats.extend(feat)
                targets.extend(target)
                
        
        mAP = cal_map(np.array(feats), np.array(targets), topK=10)
        print('>>> {}/{}  train_loss:{:.4f}, test_mAP:{:.4f}'.format(i+1, epochs, total_loss / train_count, mAP))
        if mAP > INF:
            INF = mAP
            if os.path.isfile('model.pth'):
                os.remove('model.pth')
            if len(gpu_ids) > 1:
                torch.save(model.module.state_dict(), "model.pth")
            else:
                torch.save(model.state_dict(), "model.pth")
            


if __name__ == '__main__':
    train()