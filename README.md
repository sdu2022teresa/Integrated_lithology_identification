# Integrated_lithology_identification


## Introduction

major function：

Using Pytorch to realize the dual channel image classification, it can expand the use of resnet, resnext, vgg, shuffle, mobile and other image classification networks

If it works, welcome star

## Implementation function:
* the basic function uses pytorch to realize the dual channel image classification
* using data augmentation
* add pre-training model
* add to use cnn to extract features, Softmax and other classifiers to classify
* visualization results

## Operating environment:
* python3.6.8
* torch 1.1.0
* torchvision 0.3.0
* matplotlib 3.1.0

## Quick Start

### Dataset form
 Organization form of data set:
*single polarization images: train set and test set
*orthogonal polarization images: train set and test set

### train test

Modify the parameters in 'config. py' and run 'main.py' directly


Main modified parameters:

```

dataroot  *Path to save the image
model_name  *The default is Resnet34_ DI
lr  *Learningrate
epochs  *Training total epoch
batch-size  *The size of batch
num_classes  *Number of types

```

Run 'main_test.py' to test

Modify the torch.load("*")
