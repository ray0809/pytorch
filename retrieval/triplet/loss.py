import torch.nn as nn
import torch.nn.functional as F



class TripleLoss(nn.Module):
    def __init__(self, margin):
        super(TripleLoss, self).__init__()
        self.margin = margin
        
    def forward(self, a, p, n):
        # ap = torch.sqrt(torch.sum((a - p) ** 2, dim=1) + 1e-8)
        # an = torch.sqrt(torch.sum((a - n) ** 2, dim=1) + 1e-8)
        ap = (a - p + 1e-8).pow(2).sum(1)
        an = (a - n + 1e-8).pow(2).sum(1)
        #print('55555555555', ap.shape, an.shape)
        loss = F.relu(self.margin + ap - an + 1e-8)
        return loss.mean()

