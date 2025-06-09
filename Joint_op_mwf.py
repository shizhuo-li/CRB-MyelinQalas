
import torch
import torch.nn as nn
import random
import torch.optim as optim
from mwf_func import *
from scipy.io import savemat
import math
import torch.autograd.forward_ad as fwad
import torch

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# -number of acquisitions we want to use
num_acqs = 5

# -representative tissue parameters to compute CRB for optimization
parameters = torch.tensor([[70e-3, 700e-3, 1],
                           [80e-3, 1300e-3, 1],
                           [20e-3, 130e-3, 1]])


nparam,N  = parameters.shape
tf = 128  # 128
esp = 5.74 * 1e-3  # 13.1e-3

pW = torch.tensor([1, 1, 1]).to(device)
W = torch.zeros((N, N, nparam)).to(device)
for pp in range(nparam):
    for nn in range(N):
        W[nn,nn,pp] = 1 / parameters[pp,nn]**2

input_fa = torch.ones((tf*num_acqs)) * 4 / 180 * math.pi  # used to optimize FAs
input_gap = torch.Tensor([80, 20, 58.93, 165.28, 165.28])
input_prep = torch.tensor([29.7e-3, 89.7e-3])  # used to optimize prep
input_signal = torch.cat((input_fa, input_gap, input_prep))
input_signal = input_signal.unsqueeze(0).to(device)

L = input_fa.size(0)
model = MLPWithMWF(seq_length=L, input_size=L+num_acqs+2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 0.001
loss_init = crb_loss_joint(parameters, W, pW, input_fa, input_gap, input_prep, N, num_acqs, nparam, tf, esp)
print('init_loss', loss_init)

iterations = 3
results = {'input_signal': input_signal.cpu().numpy()}
losses = []
for epoch in range(iterations):
    optimizer.zero_grad()

    fa, gap, prep = model(input_signal)

    loss_1 = crb_loss_joint(parameters, W, pW, fa, gap, prep, N, num_acqs, nparam, tf, esp)
    loss_2 = torch.sum(gap) * 40  # 初始设为20
    print(loss_1)
    print(loss_2)
    loss = loss_1 + loss_2

    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{iterations}, Loss: {loss.item()}')
    losses.append(loss.item())

    if (epoch + 1) % 250 == 0:

        fa_np = np.squeeze(fa.detach().cpu().numpy())
        gap_np = np.squeeze(gap.detach().cpu().numpy())
        prep_np = np.squeeze(prep.detach().cpu().numpy())
        output_np = np.concatenate((fa_np, gap_np, prep_np))

        results[f'output_epoch_{epoch + 1}'] = output_np

savemat('sequences/mwf_5ro_puregap.mat', results)
print(f'Saved all outputs to all_outputs.mat')

fa, gap, prep = model(input_signal)

fa_np = np.squeeze(fa.detach().cpu().numpy())
gap_np = np.squeeze(gap.detach().cpu().numpy())
prep_np = np.squeeze(prep.detach().cpu().numpy())
input_fa = input_fa.cpu().numpy().flatten()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(input_fa)
plt.title("Input fa")
plt.xlabel("Index")
plt.ylabel("Value")

# 绘制MLP输出信号
plt.subplot(1, 2, 2)
plt.plot(fa_np)
plt.title("MLP Output fa")
plt.xlabel("Index")
plt.ylabel("Value")

# 显示图像
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
