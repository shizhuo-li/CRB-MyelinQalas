import torch
import torch.nn as nn
import random
import torch.optim as optim
import torch.autograd.forward_ad as fwad
import math
from simulation_mwf import *

class MLPWithMWF(nn.Module):
    def __init__(self, input_size=626, hidden_size1=300, hidden_size2=100, num_heads=4, seq_length=640):
        super(MLPWithMWF, self).__init__()

        # 定义 MLP 部分
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 输出层
        self.fc3 = nn.Linear(hidden_size2, hidden_size1)
        self.fc4_fa = nn.Linear(hidden_size1, seq_length)
        self.fc4_gap = nn.Linear(hidden_size1, input_size-seq_length-2)
        self.fc4_prep = nn.Linear(hidden_size1, 2)

    def forward(self, x):
        # MLP 前向传播部分
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)

        # MLP 后续层
        out = self.fc3(out)
        out = self.relu(out)

        # 输出三个部分
        signal_output = self.fc4_fa(out)  # 620
        params1_output = self.fc4_gap(out)  # 5
        params2_output = self.fc4_prep(out)  # 1

        signal_output = torch.sigmoid(signal_output) * 4 / 180 * math.pi + 2 / 180 * math.pi
        params1_output = torch.sigmoid(params1_output) * 250e-3  # 0~250
        params2_output = torch.sigmoid(params2_output) * 190e-3 + 19.7e-3  # 19.7~209.7

        return signal_output, params1_output, params2_output


def crb_loss_joint(parameters, W, pW, fa, gap, prep, N, num_acqs, nparam, tf, esp):

    total_crb = 0
    fim_t = torch.zeros((N, N, nparam))
    for pp in range(nparam):
        primal = parameters[pp, :].clone().requires_grad_()
        tangs = torch.eye(N)

        fwd_jac = []

        with fwad.dual_level():

            for tang in tangs:
                dual_input = fwad.make_dual(primal, tang)
                dual_output = simulate_mwf(fa, gap, prep, dual_input, num_acqs, tf, esp)

                jacobian_column = fwad.unpack_dual(dual_output).tangent
                fwd_jac.append(jacobian_column)

        fwd_jac = torch.stack(fwd_jac).T
        fim = W[:, :, pp] @ torch.inverse(fwd_jac.T @ fwd_jac)
        fim_t[:, :, pp] = torch.real(fim)  # Show T1 T2 CRB values separately
        crb = torch.real(torch.trace(fim)) * pW[pp]

        total_crb += crb

    return total_crb
