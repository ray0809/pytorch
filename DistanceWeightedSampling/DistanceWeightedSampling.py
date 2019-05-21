
# coding: utf-8

# In[ ]:

import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
# import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset,DataLoader

import pretrainedmodels
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# # Dataloader

# In[ ]:

class BalancedBatchSampler(Sampler):
    def __init__(self, txt, batch_size, batch_k):
        assert (batch_size % batch_k == 0 ) and (batch_size > 0)
        self.dataset = {}
        self.balanced_max = 0
        self.batch_size = batch_size
        self.batch_k = batch_k

        
        # Save all the indices for all the classes
        count = 0
        with open(txt, 'r') as f:
            for i, j in enumerate(f):
                j = j.strip().split()
                if j[1] not in self.dataset:
                    self.dataset[j[1]] = []
                self.dataset[j[1]].append(i)
                count += 1
        
        num_samples = [len(value) for value in self.dataset.values()]
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)
        self.length = count // batch_size

        assert self.min_samples >= self.batch_k
    
        self.keys = list(self.dataset.keys())
        #self.currentkey = 0

    def __iter__(self):
        # while True:
        #     batch = []
        #     classes = random.sample(range(len(self.keys)), k=int(self.batch_size/self.batch_k))
        #     for cls in classes:
        #         cls_idxs = self.dataset[self.keys[cls]]
        #         batch.extend(random.sample(cls_idxs, k=self.batch_k))
        #     yield batch
        batch = []
        for i in range(self.length):
            classes = random.sample(range(len(self.keys)), k=int(self.batch_size/self.batch_k))
            for cls in classes:
                cls_idxs = self.dataset[self.keys[cls]]
                batch.extend(random.sample(cls_idxs, k=self.batch_k))
            
            yield batch
            batch = []

    def __len__(self):
        return self.length

# 测试采样器
# samp = BalancedBatchSampler('./txt/train_new.txt', 16, 4)
# samp = iter(samp)
# next(samp)


# In[ ]:


def l2n(x, eps=1e-6, dim=1):
    x = x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps).expand_as(x)
    return x


# 等价于删除某个层
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def resnet50():
    res = pretrainedmodels.resnet50(pretrained='imagenet')
    res.last_linear = Identity()  # 最后一层全连接删除
    return res
    

class MarginNet(nn.Module):
    r"""Embedding network with distance weighted sampling.
    It takes a base CNN and adds an embedding layer and a
    sampling layer at the end.
    Parameters
    ----------
    base_net : Block
        Base network.
    emb_dim : int
        Dimensionality of the embedding.
    feat_dim : int
        Dimentionality of the last linear layer of base_net.
    
    Outputs:
        - The output of DistanceWeightedSampling.
    """
    
    def __init__(self, base_net, emb_dim, feat_dim):
        super(MarginNet, self).__init__()
        self.base_net = base_net
        self.dense = nn.Linear(feat_dim, emb_dim, bias=False)
        self.normalize = l2n

    def forward(self,x):
        x = self.base_net(x)
        x = self.dense(x)
        x = self.normalize(x)
        return x


class DistanceWeightedSampling(nn.Module):
    r"""Distance weighted sampling.
    See "sampling matters in deep embedding learning" paper for details.
    Implementation similar to https://github.com/chaoyuaw/sampling_matters
    """
    def __init__(self, cutoff=0.5, nonzero_loss_cutoff=1.4):
        super().__init__()
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff

    @staticmethod
    def get_distance(x):
        """
        Helper function for margin-based loss. Return a distance matrix given a matrix.
        Returns 1 on the diagonal (prevents numerical errors)
        """
        n = x.size(0)
        square = torch.sum(x ** 2.0, dim=1, keepdim=True)
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        return torch.sqrt(distance_square + torch.eye(n, dtype=x.dtype, device=x.device))

    def forward(self, embedding, target):
        """
        Inputs:
            - embedding: embeddings of images in batch
            - target: id of instance targets
        Outputs:
            - a dict with
               * 'anchor_embeddings'
               * 'negative_embeddings'
               * 'positive_embeddings'
               with sampled embeddings corresponding to anchors, negatives, positives
        """

        distance = self.get_distance(embedding)
        distance = torch.clamp(distance, min=self.cutoff)

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(C)) * torch.log(distance)
                       - (float(C - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))
        weights = torch.exp(log_weights - log_weights.max())

        unequal = target.view(-1, 1)
        unequal = (unequal != unequal.t())

        weights = weights * (unequal & (distance < self.nonzero_loss_cutoff)).float()
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.detach().cpu().numpy()
        unequal_np = unequal.cpu().numpy()

        for i in range(B):
            same = (1 - unequal_np[i]).nonzero()[0]

            if np.isnan(np_weights[i].sum()):  # 0 samples within cutoff, sample uniformly
                np_weights_ = unequal_np[i].astype(float)
                np_weights_ /= np_weights_.sum()
            else:
                np_weights_ = np_weights[i]
            try:
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_, replace=False).tolist()
            except ValueError:  # cannot always sample without replacement
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_).tolist()

            for j in same:
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return {'anchor_embeddings': embedding[a_indices],
                'negative_embeddings': embedding[n_indices],
                'positive_embeddings': embedding[p_indices]}


class MarginLoss(nn.Module):
    r"""Margin based loss.
    Parameters
    ----------
    beta_init: float
        Initial beta
    margin : float
        Margin between positive and negative pairs.
    """
    def __init__(self, beta_init=1.2, margin=0.2):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self._margin = margin

    def forward(self, anchor_embeddings, negative_embeddings, positive_embeddings, eps=1e-8):
        """
        Inputs:
            - input_dict: 'anchor_embeddings', 'negative_embeddings', 'positive_embeddings'
        Outputs:
            - Loss.
        """

        d_ap = torch.sqrt(torch.sum((positive_embeddings - anchor_embeddings) ** 2, dim=1) + eps)
        d_an = torch.sqrt(torch.sum((negative_embeddings - anchor_embeddings) ** 2, dim=1) + eps)

        pos_loss = torch.clamp(d_ap - self.beta + self._margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self._margin, min=0.0)

        pair_cnt = float(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).item())

        # Normalize based on the number of pairs
        loss = (torch.sum(pos_loss + neg_loss)) / max(pair_cnt, 1.0)

        return loss


class SampledMarginLoss(nn.Module):
    """
    Combines DistanceWeightedSampling + Margin Loss
    """
    def __init__(self, sampling_args={}, margin_args={}):
        super().__init__()
        self.sampling = DistanceWeightedSampling(**sampling_args)
        self.margin = MarginLoss(**margin_args)

    def forward(self, embedding, target):
        sampled_dict = self.sampling(embedding, target)
        loss = self.margin(**sampled_dict)
        return loss


# In[ ]:

def adjust_learning_rate(optimizer, factor):
    """Sets the learning rate to the initial LR decayed by epoch"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


# In[ ]:

model = MarginNet(base_net=resnet50(), emb_dim=128, feat_dim=2048)
#model.load_state_dict(torch.load('resnet50_jpg_beta2.ckpt'))
model = model.to(device)

criterion = SampledMarginLoss()
criterion = criterion.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_beta = torch.optim.SGD([criterion.margin.beta], lr=0.02, momentum=0.9)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# optimizer_beta = torch.optim.Adam([criterion.margin.beta], lr=0.002)


# In[ ]:

class TripletDataset(Dataset):
    def __init__(self, txt, transforms=None):
        self.img_paths = []
        self.labels = []
        self.transforms = transforms
        
        with open(txt, 'r') as f:
            for i in f:
                i = i.strip().split()
                self.img_paths.append(i[0])
                self.labels.append(int(i[1]))
#         print(min(self.labels), max(self.labels))
                
    def __getitem__(self, idx):
        
        if self.transforms:
            img = self.transforms(self._read(self.img_paths[idx]))
        else:
            img = self._read(self.img_paths[idx])
            
        label = self.labels[idx]
        
        return img, label
            
    
    def _read(self, path):
        img = cv2.imread(path, 1)
        img = self._padding(img)
        return Image.fromarray(img.astype('uint8'))
            
            
    def _padding(self, img):
        shape = img.shape
        h, w = shape[:2]
        width = np.max([h, w])
        padd_h = (width - h) // 2
        padd_w = (width - w) // 2
        if len(shape) == 3:
            padd_tuple = ((padd_h,width-h-padd_h),(padd_w,width-w-padd_w),(0,0))
        else:
            padd_tuple = ((padd_h,width-h-padd_h),(padd_w,width-w-padd_w))
        img = np.pad(img, padd_tuple, 'constant')
        return img  
    
    def __len__(self):
        pass
    


# In[ ]:

traingen = TripletDataset('./txt/train_sub.txt', transforms=transforms.Compose([
                                                                    transforms.Resize((256,256)),
                                                                    transforms.RandomCrop(224,224),
                                                                    transforms.RandomRotation(20),
                                                                    transforms.RandomHorizontalFlip(0.5),
                                                                    transforms.ToTensor()]))
trainloader = DataLoader(dataset=traingen,
                         batch_sampler=BalancedBatchSampler('./txt/train_sub.txt', 40, 4),
                         num_workers=20,
                         pin_memory=False)


testgen = TripletDataset('./txt/test_sub.txt', transforms=transforms.Compose([transforms.Resize((256,256)),
                                                                            transforms.RandomCrop(224,224),
                                                                            transforms.ToTensor()]))
testloader = DataLoader(dataset=testgen,
                         batch_sampler=BalancedBatchSampler('./txt/test_sub.txt', 40, 2),
                         num_workers=20,
                         pin_memory=False)




# In[ ]:

epochs = 30
INF =  np.inf
for i in range(1, epochs+1):
    model.train()
    total_loss = 0
    for j, (imgs, target) in tqdm(enumerate(trainloader)):
        if j == len(trainloader):
            break
        imgs = imgs.to(device)
        target = target.to(device)
        # print(tag)
        output = model(imgs)
        loss = criterion(output, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        optimizer_beta.zero_grad()
        
        loss.backward()
        optimizer.step()
        optimizer_beta.step()
    
        
    print('############## taining ##############')
    print('{}/{}, loss:{}'.format(i+1, epochs, total_loss / len(trainloader)))
    print('############## taining ##############')
    if i % 5 == 0:
        adjust_learning_rate(optimizer, 0.8)
        adjust_learning_rate(optimizer_beta, 0.8)
    
    
    with torch.no_grad():
        model.eval()
        test_total_loss = 0
        for j, (imgs, target) in tqdm(enumerate(testloader)):
            if j == len(testloader):
                break
            imgs = imgs.to(device)
            target = target.to(device)
            # print(tag)
            output = model(imgs)
            loss = criterion(output, target)
            test_total_loss += loss.item()
            
        print('############## testing ##############')
        loss = test_total_loss / len(testloader)
        print('{}/{}, loss:{}'.format(i+1, epochs, loss))
        print('############## testing ##############')
        if loss < INF:
            INF = loss
            if os.path.isfile('res50_margin.ckpt'):
                os.remove('res50_margin.ckpt')
            torch.save(model.state_dict(),'res50_margin.ckpt')
        

