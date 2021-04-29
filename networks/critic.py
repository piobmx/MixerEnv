import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class CriticalNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticalNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.output_act = output_act

    def forward(self, state, action):
        out = F.relu(self.fc1(state))
        out = torch.cat([out, action], 1)
        out = F.relu(self.fc2(out))
        # out = self.output_act(out)
        return out


class ConCriticNet(nn.Module):
    def __init__(self):
        super(ConCriticNet, self).__init__()
        self.r1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, dilation=1)
        self.fc1 = nn.Linear(2204, 32)
        self.r2 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=2, dilation=1)
        self.fc2 = nn.Linear((32 + 1), 1)
        self.r3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, dilation=4)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        """
        Output is action (policy)
        """
        out = F.relu(self.r1(state))
        # print(f"r1 {out.shape}")
        out = F.relu(self.fc1(out))
        out = F.relu(self.r2(out))
        # print(f"out {out.shape}")
        # print(f"action {action.shape}")
        out = torch.cat([out, action], -1)
        # out = F.relu(self.fc3(out))
        # print(f"outs2 {out.shape}")
        out = self.fc3(out)
        # print(f"out3 {out.shape}")
        return out


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