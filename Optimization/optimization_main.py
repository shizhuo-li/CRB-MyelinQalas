import torch.optim as optim
from models import *
from scipy.io import savemat
import math
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
                           [20e-3, 150e-3, 1]])

nparam,N  = parameters.shape
tf = 128
mu = 40
# -initializing flipangle train with standard 4 degree flip angles
alpha = torch.ones((tf*num_acqs)) * 4 / 180 * math.pi
alpha.requires_grad = True

alpha_init = alpha.clone()

pW = torch.tensor([1, 1, 1]).to(device)


#-defining weighting matrix for each of the parameters we want to estimate
W = torch.zeros((N, N, nparam)).to(device)
for pp in range(nparam):
    for nn in range(N):
        W[nn,nn,pp] = 1 / parameters[pp,nn]**2

input_fa = torch.ones((tf*num_acqs)) * 4 / 180 * math.pi
input_gap = torch.Tensor([0.84442, 0.84442, 0.9, 0.9, 0.9])
input_prep = torch.tensor([29.7e-3, 89.7e-3])
input_signal = torch.cat((input_fa, input_gap, input_prep))
input_signal = input_signal.unsqueeze(0).to(device)
L = input_fa.size(0)

model = MLPWithCRB(seq_length=L, input_size=L+num_acqs+2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_init = crb_loss_joint(parameters, W, pW, input_fa, input_gap, input_prep, N, num_acqs, nparam)
print('init_loss', loss_init)

iterations = 1000
results = {'input_signal': input_signal.cpu().numpy()}
for epoch in range(iterations):
    optimizer.zero_grad()

    fa, gap, prep = model(input_signal)

    loss_1 = crb_loss_joint(parameters, W, pW, fa, gap, prep, N, num_acqs, nparam)
    loss_2 = torch.sum(gap) * mu
    loss = loss_1 + loss_2
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{iterations}, Loss: {loss.item()}')

    if (epoch + 1) % 250 == 0:

        fa_np = np.squeeze(fa.detach().cpu().numpy())
        gap_np = np.squeeze(gap.detach().cpu().numpy())
        prep_np = np.squeeze(prep.detach().cpu().numpy())
        output_np = np.concatenate((fa_np, gap_np, prep_np))

        results[f'paras_epoch_{epoch + 1}'] = output_np

savemat('Omni_qalas.mat', results)
print(f'Saved all outputs to all_outputs.mat')
