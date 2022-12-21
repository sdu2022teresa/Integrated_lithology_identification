import torchvision
import torch.nn.functional as F 
from torch import nn
from dataset.config import config

class densenet121(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(densenet121, self).__init__()
        self.n_classes = n_classes
        self.model = torchvision.models.densenet121(pretrained=pretrained)
        self.classifier = nn.Linear(1000, self.n_classes)
    def forward(self, x):
        features = self.model(x)
        out = F.sigmoid(self.classifier(features))
        return out

class inceptionV3(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(inceptionV3, self).__init__()
        self.n_classes = n_classes
        self.model = torchvision.models.inception_v3(pretrained=pretrained)
        self.classifier = nn.Linear(2048, self.n_classes)
    def forward(self, x):
        features = self.model(x)
        out = F.sigmoid(self.classifier(features))
        return out

class Resnet101(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(Resnet101, self).__init__()
        self.n_classes = n_classes
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        self.classifier = nn.Linear(1000, self.n_classes)
    def forward(self, x):
        features = self.model(x)
        out = F.sigmoid(self.classifier(features))
        return out