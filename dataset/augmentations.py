from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import CenterCrop

from .config import DefaultConfigs
config = DefaultConfigs()

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def get_train_transform(mean=mean, std=std, size=0):
    train_transform = transforms.Compose([
        Resize((int(size[0] * (256 / 224)), int(size[1] * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform

def get_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
