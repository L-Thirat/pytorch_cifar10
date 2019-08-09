#!/usr/bin/env python

"""
    convnet/main.py
"""

import sys
import json
import argparse
import numpy as np
from time import time

import torch
import torch.nn as nn
import os
import torch.optim as optim


import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# --
# User code
# Note: Depending on how you implement your model, you'll likely have to change the parameters of these
# functions.  They way their shown is just one possble way that the code could be structured.

criterion = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,alp=0.125):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_ch,resd_block,num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=input_ch, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-5, momentum=args.momentum, affine=False, track_running_stats=False)
        flattened_list = [y for x in resd_block for y in x]
        flattened_list = list(dict.fromkeys(flattened_list))
        self.layer1 = self._make_layer(block, flattened_list[0], num_blocks[0], stride=1,alpha=0.125)
        self.layer2 = self._make_layer(block, flattened_list[1], num_blocks[1], stride=2,alpha=0.125)
        self.layer3 = self._make_layer(block, flattened_list[2], num_blocks[2], stride=2,alpha=0.125)
        self.layer4 = self._make_layer(block, flattened_list[3], num_blocks[3], stride=2,alpha=0.125)
        self.linear = nn.Linear(flattened_list[3]*block.expansion, num_classes,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride,alpha):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,alpha))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def make_model(input_channels, output_classes, residual_block_sizes, scale_alpha, optimizer, lr, momentum):
    # ... your code here ...
    model = ResNet(BasicBlock, [2,2,2,2], input_channels,residual_block_sizes,num_classes=output_classes)
    return model


def make_train_dataloader(X, y, batch_size, shuffle):
    # ... your code here ...
    dataset = []
    for i_x,i_y in zip(X,y):
        dataset.append([torch.from_numpy(np.asarray(i_x)),i_y])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)#**
    return dataloader


def make_test_dataloader(X, batch_size, shuffle):
    # ... your code here ...
    dataset = []
    for i_x in X:
        dataset.append(torch.from_numpy(np.asarray(i_x)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)#**
    return dataloader


def train_one_epoch(model, dataloader):
    # ... your code here ...
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        learning_rate = args.lr - (batch_idx*(args.lr/(len(list(dataloader)))))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, nesterov=False)# weight_decay=5e-4
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

    return model


def predict(model, dataloader):
    # ... your code here ...

    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predict_y = outputs.max(1)
            predicted = np.append(predicted,predict_y)
    predicted = np.asarray(predicted)
    return predicted

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--num-epochs','-ne', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # --
    # IO
    
    # X_train: tensor of shape (number of train observations, number of image channels, image height, image width)
    # X_test:  tensor of shape (number of train observations, number of image channels, image height, image width)
    # y_train: vector of [0, 1] class labels for each train image
    # y_test:  vector of [0, 1] class labels for each test image (don't look at these to make predictions!)

    # X_train = np.load('X_test.npy')#**
    # X_test = np.load('X_test.npy')
    # y_train = np.load('y_test.npy')#**
    # y_test = np.load('y_test.npy')

    # X_train = np.load('../data/cifar2/X_train.npy')
    # X_test  = np.load('../data/cifar2/X_test.npy')
    # y_train = np.load('../data/cifar2/y_train.npy')
    # y_test  = np.load('../data/cifar2/y_test.npy')

    X_train = np.load('data/cifar2/X_train.npy')
    X_test  = np.load('data/cifar2/X_test.npy')
    y_train = np.load('data/cifar2/y_train.npy')
    y_test  = np.load('data/cifar2/y_test.npy')
    
    # --
    # Define model
    
    model = make_model(
        input_channels=3,
        output_classes=2,
        residual_block_sizes=[
            (16, 32),
            (32, 64),
            (64, 128),
        ],
        scale_alpha=0.125,
        optimizer="SGD",
        lr=args.lr,
        momentum=args.momentum,
    )
    model = model.to(device)

    # --
    # Train
    
    t = time()
    for epoch in range(args.num_epochs):
        
        # Train
        model = train_one_epoch(
            model=model,
            dataloader=make_train_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
        )

        # Evaluate
        preds = predict(
            model=model,
            dataloader=make_test_dataloader(X_test, batch_size=args.batch_size, shuffle=False)
        )
        # print(len(preds))
        # print(preds)
        # print(y_test)
        # check = np.ravel(preds)
        # preds = np.array(check)

        # print(check.shape)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == X_test.shape[0]
        
        test_acc = (preds == y_test.squeeze()).mean()
        
        print(json.dumps({
            "epoch"    : int(epoch),
            "test_acc" : test_acc,
            "time"     : time() - t
        }))
        sys.stdout.flush()
        
    elapsed = time() - t
    print('elapsed', elapsed, file=sys.stderr)
    
    # --
    # Save results
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/preds', preds, fmt='%d')
    open('results/elapsed', 'w').write(str(elapsed))