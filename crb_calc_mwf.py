# oding = utf-8
# -*- coding:utf-8 -*-
# oding = utf-8
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import random
import torch.optim as optim
from mwf_func import *
from scipy.io import savemat
import math
import torch.autograd.forward_ad as fwad
import torch
from scipy.io import loadmat

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# -number of acquisitions we want to use
num_acqs = 6

# -representative tissue parameters to compute CRB for optimization
parameters = torch.tensor([[70e-3, 700e-3, 1],
                           [80e-3, 1300e-3, 1],
                           [20e-3, 130e-3, 1]])
# parameters = torch.tensor([[70e-3, 700e-3, 1]])
# parameters = torch.tensor([[80e-3, 1300e-3, 1]])
# parameters = torch.tensor([[20e-3, 130e-3, 1]])


# parameters = torch.tensor([[20e-3, 130e-3, 1]])

# parameters = torch.tensor([[20e-3, 130e-3, 1]])
# [130e-3, 1300e-3, 1]])
# [500e-3, 4500e-3, 1]])
#                             #t2s t1 m0

nparam, N = parameters.shape
tf = 128
esp = 5.74 * 1e-3
# setting losses and tracking for optimization

# pW = torch.tensor([1,1,1/30]).to(device)
pW = torch.tensor([1, 1, 1]).to(device)
# -relative weighting of each representative tissue parameter

# -defining weighting matrix for each of the parameters we want to estimate
W = torch.zeros((N, N, nparam)).to(device)
for pp in range(nparam):
    for nn in range(N):
        W[nn, nn, pp] = 1 / parameters[pp, nn] ** 2

input_fa = torch.ones((tf * num_acqs)) * 4 / 180 * math.pi  # used to optimize FAs
input_gap = torch.Tensor([0.84442, 0.84442, 0.9, 0.9, 0.9, 0.9])  # , 0.9])#, 0.9])
input_prep = torch.tensor([29.7e-3, 89.7e-3])  # used to optimize prep
# data = loadmat('sequences/mwf_5ro_new_50.mat')
# # data = loadmat('sequences/mwf_joint_L20_i1000_wgw_5ro.mat')
# output_epoch_1000 = data['output_epoch_1000'].flatten()  # Flatten to 1D array
# input_prep = torch.Tensor(output_epoch_1000[-2:])  # Last two elements
# input_gap = torch.Tensor(output_epoch_1000[640:645])  # Elements 641-645 (0-based index 640-644)
# input_fa = torch.Tensor(output_epoch_1000[:640])

input_signal = torch.cat((input_fa, input_gap, input_prep))
input_signal = input_signal.unsqueeze(0).to(device)
L = input_fa.size(0)

model = MLPWithAttention(seq_length=L, input_size=L + num_acqs + 2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 0.001
loss_init= crb_loss_joint(parameters, W, pW, input_fa, input_gap, input_prep, N, num_acqs, nparam, tf, esp)
print('init_loss', loss_init)
# print(fim_t)

