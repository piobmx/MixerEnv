import random, math
from collections import namedtuple, Counter

from mixer_agent import Mixer_agent
from utilz import *
from Actions import Actions
from DQN import DQN, ReplayMemory
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



class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.output_act = output_act

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.output_act(out)
        return out


class ConActorNet(nn.Module):
    def __init__(self):
        super(ConActorNet, self).__init__()
        self.r1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, dilation=1)
        self.fc1 = nn.Linear(2204, 32)
        self.r2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=2, dilation=1)
        self.fc2 = nn.Linear(15, 1)
        self.r3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, dilation=4)

    def forward(self, input_frame):
        """
        Output is action (policy)
        """
        out = F.relu(self.r1(input_frame))
        # print(out.shape)
        out = F.relu(self.fc1(out))
        # print(out.shape)
        out = F.relu(self.r2(out))
        out = F.relu(self.fc2(out))
        # print(f"act {out.shape}")
        return out


class Actor(nn.Module):
    def __init__(self, nb_status=30, nb_actions=1, hidden1=200, hidden2=100, run_batchnorm=False, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.run_batchnorm=run_batchnorm
        # self.use_bn = use_bn or use_bn_affine
        self.bn1 = nn.BatchNorm1d(hidden1, )
        self.fc1 = nn.Linear(nb_status, hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn3 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.SELU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w, init_method)

    def forward(self, state):
        # print(state.shape)
        if self.run_batchnorm:
            out = self.relu(self.bn1(self.fc1(state)))
            out = self.relu(self.bn2(self.fc2(out)))
            out = self.relu(self.fc3(out))
        else:
            out = self.relu(self.fc1(state))
            out = self.relu(self.fc2(out))
            out = self.relu(self.fc3(out))
            # print(out.shape)
        # out = self.fc3(out)
        # out = self.tanh(out)
        return out

