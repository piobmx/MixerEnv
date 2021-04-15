from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class DQN(nn.Module):

    def __init__(self, h, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(3)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(8)

        #         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #         self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv1d_size_out1(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv1d_size_out2(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convh = conv1d_size_out2(conv1d_size_out1(h))
        # print(f" convh: {convh}")
        linear_input_size = 1 * convh * 8
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        # print(f"xshape: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        #         x = F.relu(self.bn3(self.conv3(x)))
        #         print(x.shape)
        # print(x.view(x.size(0), -1).shape)
        return self.head(x.view(x.size(0), -1))



# In[116]:


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Replay memory that is accessed to update weights
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        print(f"len(memory) {len(memory)} < batch size {BATCH_SIZE}")
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print(state_batch.dtype)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a, b, c = 100, 100, 3
    #  [16, 3, 5],
    state = torch.randn(1, 1, 4410)

    policy_net = DQN(4410, 3).to(device)
    target_net = DQN(4410, 3).to(device)
    target_net.eval()


    output = policy_net(state)
    print(f"shape: {output.shape}")
    # print(output.max(1)[1])
    print(output.max(1)[1].view(1, 1))

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    for x in range(10000):
        if x % 2000 == 0: print(f"pushing {x}")
        state = torch.randint(0, 100, (1, 1, 4410), dtype=torch.float32)
        next_state = torch.randint(0, 1000, (1, 1, 4410), dtype=torch.float32)
        action = torch.randint(0, 3, (1, 1))
        print(action.shape)
        # reward = torch.randint(0, 1, (1, 1)) / 10
        reward = np.random.random()
        reward = torch.tensor([reward])

        memory.push(state, action, next_state, reward)


    for y in range(100):
        optimize_model()
