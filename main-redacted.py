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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-5, momentum=None, affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes,bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
    model = ResNet(BasicBlock, [2,2,2,2])
    return model


def make_train_dataloader(X, y, batch_size, shuffle):
    # ... your code here ...
    dataset = []
    for i_x,i_y in zip(X,y):
        dataset.append([torch.from_numpy(np.asarray(i_x)),i_y])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)#**
    # asd
    #
    # dataloader = []
    # indices = np.arange(len(X))
    # i = 0
    # x_arr=[]
    # y_arr=[]
    # while i<batch_size:
    #     if shuffle:
    #         np.random.shuffle(indices)
    #     for i in indices:
    #         dataloader.append([torch.from_numpy(X[i]),y])
    #         # x_arr.append(X[i])
    #         # y_arr.append(y[i])
    #     i+=1
    # # X_tensor = torch.FloatTensor(x_arr)
    # # y_tensor = torch.FloatTensor(y_arr)
    # # dataloader = [X_tensor,y_tensor]

    return dataloader


def make_test_dataloader(X, batch_size, shuffle):
    # ... your code here ...
    dataloader = []
    indices = np.arange(len(X))
    i = 0
    x_arr=[]
    y_arr=[]
    while i<batch_size:
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            x_arr.append(X[i])
        i+=1
    X_tensor = torch.FloatTensor(x_arr)
    dataloader = X_tensor
    return dataloader


def train_one_epoch(model, dataloader):
    # ... your code here ...
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False)# weight_decay=5e-4
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
    with torch.no_grad():
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = inputs.to(device)
            predictions = model(inputs)

    return predictions

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--num-epochs', type=int, default=5)
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

    # X_train = np.load('X_train.npy')#**
    # X_test = np.load('X_test.npy')
    # y_train = np.load('y_train.npy')#**
    # y_test = np.load('y_test.npy')

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