import random, math
from collections import namedtuple, Counter

from mixer_agent import Mixer_agent
from utilz import *
from Actions import Actions
# from DQN import DQN, ReplayMemory
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable



# class Actor(nn.Module):
#     def __init__(self, nb_status=30, nb_actions=1, hidden1=200, hidden2=100, run_batchnorm=False, learning_rate = 3e-4):
#         super(Actor, self).__init__()
#         self.run_batchnorm=run_batchnorm
#         # self.use_bn = use_bn or use_bn_affine
#         self.bn1 = nn.BatchNorm1d(hidden1, )
#         self.fc1 = nn.Linear(nb_status, hidden1)
#         self.bn2 = nn.BatchNorm1d(hidden2)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.bn3 = nn.BatchNorm1d(hidden2)
#         self.fc3 = nn.Linear(hidden2, nb_actions)
#         self.relu = nn.SELU()
#         self.tanh = nn.Tanh()
#         # self.init_weights(init_w, init_method)
#
#     def forward(self, state):
#         # print(state.shape)
#         if self.run_batchnorm:
#             out = self.relu(self.bn1(self.fc1(state)))
#             out = self.relu(self.bn2(self.fc2(out)))
#             out = self.relu(self.fc3(out))
#         else:
#             out = self.relu(self.fc1(state))
#             out = self.relu(self.fc2(out))
#             out = self.relu(self.fc3(out))
#             # print(out.shape)
#         # out = self.fc3(out)
#         # out = self.tanh(out)
#         return out

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    """
    The Actor chooses the action when receives the state. The output action should be infinite ranging from -1 to 1.
    """
    def __init__(self, n_input=2, n_output=1, stride=4, n_channel=10):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride, dilation=4)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=25, stride=1, padding=1, dilation=8)
        self.bn2 = nn.BatchNorm1d(2 * n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.avgpool = nn.AvgPool1d(1)
        self.dense1 = nn.Linear(320, 32)
        # self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.relu = torch.relu
        self.output_ = torch.tanh

        self.dense2 = nn.Linear(32, n_output)
        self.init_weights()

    def forward(self, x):
        # print(x.shape)
        if True in torch.isnan(x):
            x = torch.nan_to_num(x, nan=np.random.random())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(self.pool1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(self.pool2(x))
        # print(x.shape)
        # x = F.avg_pool1d(x, x.shape[-1])
        x = self.avgpool(x)
        x = self.bn2(x)
        # if True in torch.isnan(x):
        #     print("TRUE HERE")
        # x = x.permute(0, 2, 1)
        # if x.size > 20000:
        x = x.view(-1, 320)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_(x)
        return x

    def init_weights(self, init_w=0.0001):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        # self.fc3.weight.data.uniform_(-init_w, init_w)