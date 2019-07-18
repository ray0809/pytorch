import torch
import torch.nn as nn

import numpy as np 

class Margin_Loss(nn.Module):
    """ the margin losss contain the distane weighted sampling and margin based loss,
    sampling and margin loss compute based on paper 'Sampling Matters in Deep Embedding Learning' """
    def __init__(self,batch_k=5,margin=0.2,nu=0.0,cutoff=0.5,nonzero_loss_cutoff=1.4):
        """
        this loss function receive batch of image_feature,then compute the distance weighted sampling loss
        :param batch_k: images count for every class
        :param margin: margin for alpha in paper
        :param nu: regularization parameter for beta
        """
        super(Margin_Loss,self).__init__()
        self.margin = torch.tensor(margin,dtype=torch.float32).cuda()
        self.nu = torch.tensor(nu,dtype=torch.float32).cuda()
        self.batch_k = batch_k
        self.cutoff = cutoff                           # to cut for probbality
        self.nonzero_loss_cutoff = nonzero_loss_cutoff # to cut the distance upper bound
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()





    def forward(self,x,y,beta_in):
        """
        :param x: x is the feature extracted from resnet,data type torch.tensor,data.shape (n.d) typical (70,128)
        :param y: the label for each small class range from 0-200,so as the same dimension of beta_in
        :param beta_in: beta_in is a torch variable (tensor) with require_grad = True
        :return: the loss of beta_reg_loss and margin loss
        """
        a_index,p_index,n_index = self.sampling(x) # so the corresponding anchor,postive and negitve has belong the distance weighted distribution
        beta_work =  beta_in[a_index] # get the coeffient of the beta data
        beta_reg_loss = torch.sum(beta_work)*self.nu   # loss batckward valid

        # compute margin loss from feature
        anchors = x[a_index]
        postives = x[p_index]
        negtives = x[n_index]
        d_ap = torch.sqrt(torch.sum((anchors - postives)*(anchors - postives),1)+1e-8)
        d_an = torch.sqrt(torch.sum((anchors - negtives)*(anchors - negtives),1)+1e-8)

        pos_loss = self.relu1(d_ap - beta_work + self.margin)
        neg_loss = self.relu2(beta_work - d_an + self.margin)
        pair_cnt = torch.sum((pos_loss>0.0) +(neg_loss>0.0))

        # normalize based on the number of pairs
        loss = (torch.sum(pos_loss + neg_loss) + beta_reg_loss)/ pair_cnt.float()#pair_cnt.numpy()[0]
        return loss


    def sampling(self,x):
        """
        sampling images pairs based on distance of each images
        :param x: x is the [N,128] tensor of the extracted features
        :return: anchors,postives,negtives
        """
        np_feature = x.detach().cpu().numpy()
        k = self.batch_k
        n,d = np_feature.shape

        # compute distance
        dis_matrix = self.get_distance(np_feature)

        # cut off to avoid hight variance
        dis_matrix = np.maximum(dis_matrix,self.cutoff)

        log_weights = ((2.0 - float(d)) * np.log(dis_matrix)
                       - (float(d-3)/2)*np.log(1.0-0.25*(dis_matrix**2)))
        #weights = np.exp(log_weights - log_weights.max(1).reshape(-1,1)) #log_weights-log_weights.max(1).reshape(-1,1), every line subtract the max weight number ,not the total number
        weights = np.exp(log_weights - log_weights.max())
        mask = np.ones(weights.shape)
        for i in range(0,n,k):
            mask[i:i+k,i:i+k] = 0 # to set block in indentity line surrounding box is 0

        weights = weights * mask *(dis_matrix<self.nonzero_loss_cutoff)
        weights = weights / (weights.sum(axis=1,keepdims=True) + 1e-8)

        a_index ,p_index,n_index = [],[],[]
        for i in range(n):
            block_idx = i//k    # k is self.batch_k,typical is 5
            try:
                n_index += np.random.choice(n,k-1,p= weights[i]).tolist()
            except:
                n_index += np.random.choice(n,k-1).tolist()
            for j in range(block_idx*k,(block_idx+1)*k):
                if j != i:
                    a_index.append(i)
                    p_index.append(j)

        return a_index,p_index,n_index

    def get_distance(self,x):
        """
        compute the distance between every two feature vector, construct a distance matrix shaped [N,N] for same vecotr the distance set 1 instead
        :param x: feature vector of N samples
        :return: distance of two feature,set 1 for same vector
        """
        n = x.shape[0]
        square = np.sum(x*x,axis=1,keepdims = True)
        distance_square = square + square.transpose() - (2*np.matmul(x,x.transpose()))
        distance_identity = distance_square + np.identity(n)
        return np.sqrt(distance_identity)