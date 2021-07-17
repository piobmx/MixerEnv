import numpy as np
import torch
import torch.nn as nn


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class actor_mini(nn.Module):
    """
    The Actor chooses the action when receives the state. The output action should be infinite ranging from -1 to 1.
    """
    def __init__(self, input_size=3, hidden_size_1=64, hidden_size_2=64, action_dim=1, output_size=1, norm=False):
        super(actor_mini, self).__init__()
        self.norm = norm
        assert output_size == action_dim
        self.dense1 = nn.Linear(input_size, hidden_size_1)
        self.dense2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.dense3 = nn.Linear(hidden_size_2, output_size)
        self.bn1 = nn.BatchNorm1d(num_features=1)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        self.bn3 = nn.BatchNorm1d(num_features=1)
        self.relu = torch.relu
        self.output_ = torch.tanh
        self.init_weights()

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        if self.norm:
            x = self.bn1(self.relu(self.dense1(x)))
            x = self.bn2(self.relu(self.dense2(x)))
            x = self.relu(self.dense3(x))
            # x = self.dense3(x)
        if not self.norm:
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            x = self.dense3(x)
        x = self.output_(x)
        return x

    def init_weights(self):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense3.weight.data = fanin_init(self.dense3.weight.data.size())
        self.dense1.bias.data = fanin_init(self.dense1.bias.data.size())
        self.dense2.bias.data = fanin_init(self.dense2.bias.data.size())
        self.dense3.bias.data = fanin_init(self.dense3.bias.data.size())


class MelActor(nn.Module):
    def __init__(self, hidden_size_1=32, hidden_size_2=64, stride=4):
        super().__init__()
        self.conv1 = nn.Conv2d(2, hidden_size_1, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(hidden_size_2)

        self.dense1 = nn.Linear(hidden_size_2 * 31, hidden_size_2)
        self.dense2 = nn.Linear(hidden_size_2, 1)
        self.activ = torch.relu
        self.output_ = torch.tanh
        self.init_weights()

    def forward(self, s):
        s = self.conv1(s)
        s = self.bn1(s)
        s = self.conv2(s)
        s = self.bn2(s)
        s = s.view(-1, 1, 64 * 31)
        s = self.activ(self.dense1(s))
        s = self.dense2(s)
        out = self.output_(s)
        return out

    def init_weights(self):
        self.dense1.weight.data = fanin_init(self.dense1.weight.data.size())
        self.dense2.weight.data = fanin_init(self.dense2.weight.data.size())
        self.dense1.bias.data = fanin_init(self.dense1.bias.data.size())
        self.dense2.bias.data = fanin_init(self.dense2.bias.data.size())