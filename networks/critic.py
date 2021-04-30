import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Critic(nn.Module):
    def __init__(self, nb_status, nb_actions, hidden1=100, hidden2=40):
        super(Critic, self).__init__()
        # self.use_bn = use_bn or use_bn_affine
        self.fcs = nn.Linear(nb_status, hidden1)
        self.fca = nn.Linear(nb_actions, hidden1)

        self.bns = nn.BatchNorm1d(hidden1)
        self.bna = nn.BatchNorm1d(hidden1)

        self.fc1 = nn.Linear(hidden1 + 1, hidden1 // 2)
        self.bn1 = nn.BatchNorm1d(hidden1 // 2, )
        self.fc2 = nn.Linear(hidden1 // 2, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, )
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w, init_method)

    def forward(self, state_action):
        s, a = state_action
        s = self.fcs(s)
        # s = self.relu(self.bns(self.fcs(s)))
        a = self.relu(a)
        #         a = self.relu(self.bna(self.fca(a)))
        # print(f"state shape: {s.shape}")
        # print(f"action shape: {a.shape}")
        # print(torch.cat([s, a], 1).shape)
        out = self.fc1(torch.cat([s, a], 2))
        # out = self.bn1(out)
        out = self.fc2(out)
        # out = self.bn2(self.fc2(out))
        out = self.fc3(out)
        out = self.relu(out)
        return out
