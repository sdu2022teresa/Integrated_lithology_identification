import torch.utils.data as data
from PIL import Image
import cv2
import os
import torch
import torchvision.transforms.functional as tf
from torchvision import transforms
import random
import numpy as np

#Single input
def make_dataset(imgroot, data = 'train'):
    imgs = []
    # assert len(os.listdir(img1root)) == len(os.listdir(segroot))
    # assert len(os.listdir(img2root)) == len(os.listdir(segroot))
    for dir in os.listdir(imgroot +'/'+ data):
        for img in os.listdir(imgroot +'/'+ data + '/' + dir):
            imgpath = imgroot +'/'+ data + '/' + dir + '/' + img
            imgs.append((imgpath, int(dir)))
    return imgs

class myDataset(data.Dataset):
    def __init__(self, imgroot, width, height, n_class, transform=None, target_transform = True, data = 'train'):
        imgs = make_dataset_DI(imgroot , data)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.w = width
        self.h = height
        self.classes = n_class
        self.data = data

    def rotate(self, image1, angle=None):
        if random.random() > 0.5:
            if angle == None:
                angle = transforms.RandomRotation.get_params([-180, 180])
            if isinstance(angle, list):
                angle = random.choice(angle)
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image1 = cv2.warpAffine(image1, M, (self.w, self.h))
            # image2 = cv2.warpAffine(image2, M, (self.w, self.h))
            # image = tf.to_tensor(image)
            # mask = tf.to_tensor(mask)
        return image1

    def flip(self, image1):  # Flip horizontal and Flip vertical
        if random.random() > 0.5:
            image1 = cv2.flip(image1, 1)
            # image2 = cv2.flip(image2, 1)
        if random.random() < 0.5:
            image1 = cv2.flip(image1, 0)
            # image2 = cv2.flip(image2, 0)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image1

    def seg_onhot(self, lable):
        seg = np.array(lable)
        seg_labels = np.eye(self.classes)[seg]
        # return tf.to_tensor(seg_labels)
        return torch.tensor(seg_labels)

    def __getitem__(self, index):
        x1_path, x2_path, lable = self.imgs[index]
        img1_x = cv2.imread(x1_path, -1)
        # img2_x = cv2.imread(x2_path, -1)
        img1_x = cv2.resize(img1_x, (self.w, self.h))
        # img2_x = cv2.resize(img2_x, (self.w, self.h))
        if self.data == 'train':
            img1_x = self.rotate(img1_x, angle=[90, 180, -90])
            img1_x = self.flip(img1_x)
        if self.transform is not None:
            img1_x = Image.fromarray(img1_x)
            img1_x = self.transform(img1_x)
            # img2_x = self.transform(img2_x)
        if self.target_transform == True:
            lable = self.seg_onhot(lable)
        return {'image1': img1_x, 'label': lable}

    def __len__(self):
        return len(self.imgs)

#Dual input
def make_dataset_DI(imgroot, data = 'train'):
    imgs = []
    # assert len(os.listdir(img1root)) == len(os.listdir(segroot))
    # assert len(os.listdir(img2root)) == len(os.listdir(segroot))
    for dir in os.listdir(imgroot +'/'+ data + '/' + 'crossed'):
        for img in os.listdir(imgroot +'/'+ data + '/' + 'crossed' +'/'+dir):
            img1path = imgroot +'/'+ data + '/' + 'crossed' +'/' + dir + '/' + img
            img2path = imgroot +'/'+ data + '/' + 'single' +'/' + dir + '/' + img.split('_')[0] + '_single (' + img.split('(')[1]
            imgs.append((img1path, img2path , int(dir)))
    return imgs

class myDataset_DI(data.Dataset):
    def __init__(self, imgroot, width, height, n_class, transform=None, target_transform = True, data = 'train'):
        imgs = make_dataset_DI(imgroot , data)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.w = width
        self.h = height
        self.classes = n_class
        self.data = data

    def rotate(self, image1, image2, angle=None):
        if random.random() > 0.5:
            if angle == None:
                angle = transforms.RandomRotation.get_params([-180, 180])
            if isinstance(angle, list):
                angle = random.choice(angle)
            center = (self.w // 2, self.h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image1 = cv2.warpAffine(image1, M, (self.w, self.h))
            image2 = cv2.warpAffine(image2, M, (self.w, self.h))
            # image = tf.to_tensor(image)
            # mask = tf.to_tensor(mask)
        return image1, image2

    def flip(self, image1, image2):
        if random.random() > 0.5:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
        if random.random() < 0.5:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
        # image = tf.to_tensor(image)
        # mask = tf.to_tensor(mask)
        return image1, image2

    def seg_onhot(self, lable):
        seg = np.array(lable)
        seg_labels = np.eye(self.classes)[seg]
        # print(seg_labels)
        # return tf.to_tensor(seg_labels)
        return torch.tensor(seg_labels)
    def __getitem__(self, index):
        x1_path, x2_path, lable = self.imgs[index]
        img1_x = cv2.imread(x1_path, -1)
        img2_x = cv2.imread(x2_path, -1)
        img1_x = cv2.resize(img1_x, (self.w, self.h))
        img2_x = cv2.resize(img2_x, (self.w, self.h))
        if self.data == 'train':
            img1_x, img2_x = self.rotate(img1_x, img2_x, angle=[90, 180, -90])
            img1_x, img2_x = self.flip(img1_x, img2_x)
        if self.transform is not None:
            img1_x = Image.fromarray(img1_x)
            img2_x = Image.fromarray(img2_x)
            img1_x = self.transform(img1_x)
            img2_x = self.transform(img2_x)
        if self.target_transform == True:
            lable = self.seg_onhot(lable)
        return {'image1': img1_x, 'image2': img2_x, 'label': lable}
        # return img1_x, img2_x, lable
    def __len__(self):
        return len(self.imgs)