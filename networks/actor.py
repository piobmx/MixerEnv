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

