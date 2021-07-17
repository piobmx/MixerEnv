import torch
import torch.nn as nn
import numpy as np


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class critic_mini(nn.Module):
    """
    The Critic receives [state, action] pair and outputs the pair's quality (good/bad)
    """
    def __init__(self, input_size=3, action_dim=1, hidden_size_1=64, hidden_size_2=64, output_size=2, norm=False):
        super(critic_mini, self).__init__()
        self.norm = norm
        concat_size = action_dim + input_size
        self.sdense = nn.Linear(input_size, 120)
        self.adense = nn.Linear(action_dim, 8)
        # concat_size = 128
        self.dense1 = nn.Linear(concat_size, hidden_size_1)
        self.dense2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.dense3 = nn.Linear(hidden_size_2, output_size)
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)
        self.activ = torch.relu
        self.output_ = torch.tanh
        self.init_weights()

    def forward(self, state_action):
        s, a = state_action
        s = s.view(s.shape[0], 1, -1)
        # s = self.bn1(self.activ(self.sdense(s)))
        # a = self.bn1(self.activ(self.adense(a)))
        out = torch.cat([s, a], 2)
        # print(s.shape, a.shape, out.shape)
        if self.norm:
            out = self.bn1(self.activ(self.dense1(out)))
            out = self.bn2(self.activ(self.dense2(out)))
            out = self.activ(self.dense3(out))
        if not self.norm:
            out = self.activ(self.dense1(out))
            out = self.activ(self.dense2(out))
            out = self.dense3(out)
        out = self.output_(out)
        return out

    def init_weights(self, init_w=0.001):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense3.weight.data = fanin_init(self.dense3.weight.data.size())
        self.sdense.weight.data = fanin_init(self.sdense.weight.data.size())
        self.adense.weight.data = fanin_init(self.adense.weight.data.size())
        self.dense1.bias.data = fanin_init(self.dense1.bias.data.size())
        self.dense2.bias.data = fanin_init(self.dense2.bias.data.size())
        self.dense3.bias.data = fanin_init(self.dense3.bias.data.size())


class MelCritic(nn.Module):
    def __init__(self, hidden_size_1=32, hidden_size_2=64, stride=4):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden_size_1, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(hidden_size_2)
        self.cdense = nn.Linear(hidden_size_2 * 31, 31)
        self.dense1 = nn.Linear(32, 8)
        self.dense2 = nn.Linear(8, 1)
        self.activ = torch.relu
        self.output_ = torch.tanh
        self.init_weights()

    def forward(self, state_action):
        s, a = state_action
        s = self.conv1(s)
        s = self.bn1(s)
        s = self.conv2(s)
        s = self.bn2(s)
        s = s.view(-1, 1, 64 * 31)
        s = self.activ(self.cdense(s))
        # print(f"S{s.shape}")
        # print(f"A{a.shape}")
        out = torch.cat([s, a], dim=2)
        out = self.activ(self.dense1(out))
        out = self.dense2(out)
        out = self.output_(out)
        return out

    def init_weights(self, init_w=0.001):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.cdense.weight.data = fanin_init(self.cdense.weight.data.size())
        self.dense1.bias.data = fanin_init(self.dense1.bias.data.size())
        self.dense2.bias.data = fanin_init(self.dense2.bias.data.size())
        self.cdense.bias.data = fanin_init(self.cdense.bias.data.size())