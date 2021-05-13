import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


#
# class Critic(nn.Module):
#     def __init__(self, nb_status, nb_actions, hidden1=100, hidden2=40):
#         super(Critic, self).__init__()
#         # self.use_bn = use_bn or use_bn_affine
#         self.fcs = nn.Linear(nb_status, hidden1)
#         self.fca = nn.Linear(nb_actions, hidden1)
#
#         self.bns = nn.BatchNorm1d(hidden1)
#         self.bna = nn.BatchNorm1d(hidden1)
#
#         self.fc1 = nn.Linear(hidden1 + 1, hidden1 // 2)
#         self.bn1 = nn.BatchNorm1d(hidden1 // 2, )
#         self.fc2 = nn.Linear(hidden1 // 2, hidden2)
#         self.bn2 = nn.BatchNorm1d(hidden2, )
#         self.fc3 = nn.Linear(hidden2, 1)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         # self.init_weights(init_w, init_method)
#
#     def forward(self, state_action):
#         s, a = state_action
#         s = self.fcs(s)
#         # s = self.relu(self.bns(self.fcs(s)))
#         a = self.relu(a)
#         #         a = self.relu(self.bna(self.fca(a)))
#         # print(f"state shape: {s.shape}")
#         # print(f"action shape: {a.shape}")
#         # print(torch.cat([s, a], 2).shape)
#         out = self.fc1(torch.cat([s, a], 2))
#         # out = self.bn1(out)
#         out = self.fc2(out)
#         # out = self.bn2(self.fc2(out))
#         out = self.fc3(out)
#         out = self.relu(out)
#         return out
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    """
    The Critic receives [state, action] pair and outputs the pair's quality (good/bad)
    """
    def __init__(self, hidden=32, n_input=2, n_output=35, stride=4, n_channel=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(n_channel)
        self.conv2 = nn.Conv1d(n_channel, n_channel // 2, kernel_size=25, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_channel // 2)
        self.pool2 = nn.MaxPool1d(n_channel // 2)
        self.conv0 = nn.Conv1d(n_input, 1, kernel_size=80, stride=stride, dilation=4)
        self.bn = nn.BatchNorm1d(1)
        self.pool = nn.MaxPool1d(1)

        self.fca = nn.Linear(1, 20)
        self.dense1 = nn.Linear(1044, 32)

        self.dense2 = nn.Linear(32, 8)
        self.dense3 = nn.Linear(8, 1)
        self.activ = torch.relu
        self.output_ = torch.tanh
        self.init_weights()

    def forward(self, state_action):
        s, a = state_action
        # print(f"s: {s.shape}")
        # print(f"a: {a.shape}")
        s = self.conv0(s)
        s = self.bn(s)
        s = self.pool(s)
        s = self.activ(s)
        a = self.fca(a)
        a = self.activ(a)
        out = torch.cat([s, a], 2)
        # print(out.shape)
        out = self.activ(self.dense1(out))
        out = self.activ(self.dense2(out))
        out = self.activ(self.dense3(out))
        out = self.output_(out)
        #         out = F.log_softmax(out, dim=2)
        return out


    def init_weights(self, init_w=3e-3):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense3.weight.data.uniform_(-init_w, init_w)