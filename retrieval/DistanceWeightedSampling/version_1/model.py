from torchvision import models
import torch.nn as nn
import torch.nn.functional as F



class MarginNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        res = models.resnet34(pretrained=True)
        emebdding = nn.Sequential(
            nn.Linear(512, embedding_dim),
        )
        res.fc = emebdding
        self.res = res

    def forward(self, x):
        x = self.res(x)
        return F.normalize(x)

    def extrct(self, x):
        x = self.res(x)
        return x