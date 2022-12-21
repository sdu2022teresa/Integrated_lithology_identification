import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from models.position_attention import *
from dataset.config import config

#     return model

class Resnet34_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(Resnet34_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.resnet34(pretrained=pretrained)
        self.model2 = torchvision.models.resnet34(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class vgg16_bn_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(vgg16_bn_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.model2 = torchvision.models.vgg16_bn(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class shufflenet_v2_x1_0_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(shufflenet_v2_x1_0_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        self.model2 = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class shufflenet_v2_x0_5_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(shufflenet_v2_x0_5_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
        self.model2 = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class shufflenet_v2_x1_5_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(shufflenet_v2_x1_5_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.shufflenet_v2_x1_5(pretrained=pretrained)
        self.model2 = torchvision.models.shufflenet_v2_x1_5(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class shufflenet_v2_x2_0_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(shufflenet_v2_x2_0_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.shufflenet_v2_x2_0(pretrained=pretrained)
        self.model2 = torchvision.models.shufflenet_v2_x2_0(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class resnext50_32x4d_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(resnext50_32x4d_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        self.model2 = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class resnext101_32x8d_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(resnext101_32x8d_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        self.model2 = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class squeezenet1_0_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(squeezenet1_0_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.squeezenet1_0(pretrained=pretrained)
        self.model2 = torchvision.models.squeezenet1_0(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class mobilenet_v2_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(mobilenet_v2_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.model2 = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class resnet18_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(resnet18_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.resnet18(pretrained=pretrained)
        self.model2 = torchvision.models.resnet18(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

class resnet50_DI(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(resnet50_DI, self).__init__()
        self.n_classes = n_classes
        self.model1 = torchvision.models.resnet50(pretrained=pretrained)
        self.model2 = torchvision.models.resnet50(pretrained=pretrained)
        self.classifier = nn.Linear(2000, self.n_classes)
    def forward(self, x1, x2):
        features1 = self.model1(x1)
        features2 = self.model2(x2)
        features = torch.cat((features1, features2), dim=1)
        out = F.sigmoid(self.classifier(features))
        return out

if __name__ == '__main__':
    height = 512
    width = 512
    # model = Res_Attn_net(n_classes=30)
    model = shufflenet_v2_x0_5_DI(n_classes=30)
    x1 = torch.randn((1, 3, height, width))
    x2 = torch.randn((1, 3, height, width))
    out = model(x1, x2)
    print(out.shape)