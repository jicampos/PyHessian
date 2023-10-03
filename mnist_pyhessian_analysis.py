#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

from utils import *
from density_plot import get_esd_plot
from models.resnet import resnet
from pyhessian import hessian

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#DB4437", "#4285F4", "#F4B400", "#0F9D58", "purple", "goldenrod", "peru", "coral","turquoise",'gray','navy','m','darkgreen','fuchsia','steelblue'])


##############################
# Define model
##############################
class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')
parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')
parser.add_argument('--data',
                    type=str,
                    default='',
                    help='path to dataset')
parser.add_argument('--config',
                    type=str,
                    default='',
                    help='config for selecting dataset features')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('../data', train=False, transform=transform)

train_kwargs = {'batch_size': args.mini_hessian_batch_size}
test_kwargs = {'batch_size': args.mini_hessian_batch_size}

train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (50000 % args.hessian_batch_size == 0) 
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

model = MNIST()
criterion = nn.NLLLoss()

###################
# Get model checkpoint, get saving folder
###################
if args.resume == '':
    raise Exception("please choose the trained model")

model.load_state_dict(torch.load(args.resume))

######################################################
# Begin the computation
######################################################
# turn model to eval mode
model.eval()
if batch_num == 1:
    hessian_comp = hessian(model,
                           criterion, 
                           data=hessian_dataloader,
                           cuda=args.cuda)
else:
    hessian_comp = hessian(model,
                           criterion,
                           dataloader=hessian_dataloader,
                           cuda=args.cuda)

print('********** finish data londing and begin Hessian computation **********')

top_eigenvalues, top_eigenvector, eigenvalueL, eigenvectorL = hessian_comp.eigenvalues()
trace, traceL = hessian_comp.trace()
density_eigen, density_weight = hessian_comp.density()

plt.figure(figsize=(12,12))
traces = [np.mean(trace_vhv) for trace_vhv in traceL.values()]
plt.plot(traces, 'o-')
plt.xlabel('Layers')
plt.ylabel('Average Hessian Trace')
plt.xticks(list(range(len(traces))), hessian_comp.layers)
plt.grid()
plt.savefig('trace-mnist-v2.png')
plt.yscale('log')
plt.savefig('trace-mnist-log-v2.png')
plt.close()

print('\n***Top Eigenvalues: ', top_eigenvalues)
print('\n***Trace: ', np.mean(trace))

print('##############################################')
print(traces)
print('##############################################')
