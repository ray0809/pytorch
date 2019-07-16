import torch.nn as nn
import torchvision


def load_resnet(num_classes):
    model = torchvision.models.resnet34(pretrained=True)

    

    classifier = nn.Sequential(
        nn.Linear(512, num_classes),
    )

    model.fc = classifier

    return model
