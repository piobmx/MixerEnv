# from actor import *
import pickle

from networks.critic import ConCriticNet, Critic
from networks.actor import ConActorNet, Actor
from networks.Memory import ReplayMemory, Transition
from networks.Evaluator import Evaluator
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


class AudioCNN(nn.Module):
    def __init__(self, num_inputs=1, num_outputs=30):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_inputs, 8, 5, stride=1, padding=2, dilation=4)
        self.maxp1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(8, 16, 5, stride=1, padding=1, dilation=8)
        self.maxp2 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(16, 16, 4, stride=1, padding=1, dilation=16)
        self.maxp3 = nn.MaxPool1d(2, 2)
        self.conv4 = nn.Conv1d(16, 32, 3, stride=1, padding=1, dilation=64)
        self.maxp4 = nn.MaxPool1d(2, 2)

        self.out = nn.Linear(6272, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        # print(f"cnn view: {x.shape}")
        x = self.out(x)
        return x


class DDPG:
    def __init__(self, mixerEnv, state_dim=0, action_dim=1, memory_size=20000, actor_output_act=F.tanh, actor_lr=0.001,
                 critic_lr=0.001, optimizer="adam", gamma=0.99, action_hidden_size=32, critic_hidden_size=32,
                 batch_size=50, eps_start=0.95, eps_end=0.01, target_tau=0.01, use_cuda=False):
        self.mixer = mixerEnv
        self.gamma, self.eps_start, self.eps_end = gamma, eps_start, eps_end
        self.target_tau = target_tau
        self.n_episode = 0
        self.use_cuda = use_cuda
        self.evaluator = None

        self.cnn = AudioCNN()
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=1e-4)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_hidden_size = action_hidden_size
        self.c_hidden_size = critic_hidden_size
        self.a_output_act = actor_output_act
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size

        self.memory = ReplayMemory(capacity=memory_size)
        # self.actor = ActorNet(self.state_dim, self.a_hidden_size, output_size=self.action_dim,
                              # output_act=self.a_output_act)
        # self.actor = ConActorNet()
        # self.critic = ConCriticNet()
        # self.critic = CriticalNet(self.state_dim, self.action_dim, self.c_hidden_size, output_size=1)
        self.actor = Actor(nb_status=30, nb_actions=1)
        self.critic = Critic(nb_status=30, nb_actions=1)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        if optimizer == "adam":
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def update_episode(self):
        self.n_episode += 1
        return

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def interact(self, epoch):
        reward = self.mixer.update_one_step()
        self.evaluator.reward_history[epoch].append(reward)

    def train(self):
        if (len(self.memory) < self.batch_size):
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_tsr = to_tensor_var(batch.state).unsqueeze(1)
        action_tsr = to_tensor_var(batch.action)
        action_batch = action_tsr.unsqueeze(1)

        reward_tsr = to_tensor_var(batch.reward)
        reward_batch = reward_tsr.unsqueeze(1)
        next_state_tsr = to_tensor_var(batch.next_state).unsqueeze(1)
        # print(f"state_tsr: {state_tsr.shape}\n"
        #       f"action_tsr: {action_batch.shape}\n"
        #       f"reward_tsr: {reward_batch.shape}\n"
        #       f"next_state_tsr:{next_state_tsr.shape}")


        # state_tsr = to_tensor_var(np.array([standardize(s) for s in state_tsr]))
        # next_state_tsr = to_tensor_var(np.array([standardize(s) for s in next_state_tsr]))
        next_state_tsr = standardize(next_state_tsr)
        state_tsr = standardize(state_tsr)

        state_batch = self.cnn(state_tsr).unsqueeze(1)
        next_state_batch = self.cnn(next_state_tsr).unsqueeze(1)
        # print(f"nsb: {next_state_batch.shape}")
        next_state_batch_act = self.actor(next_state_batch)
        # next_state_batch_act = next_state_batch_act
        # print(f"sba: {state_batch.shape}")
        # print(f"nsba: {next_state_batch_act.shape}")
        future_q_vals = self.critic_target([state_batch, next_state_batch_act])

        # next_action_tsr = self.actor_target(next_state_tsr)
        # print(f"future {future_q_vals.shape}")
        reward_batch = reward_batch.unsqueeze(1)
        # print(f"reward_shape: {reward_batch.shape}")

        target_q_batch = to_tensor_var(reward_batch) + self.gamma * future_q_vals

        # next_q = self.critic_target(next_state_tsr, next_action_tsr).detach()
        # target_q = reward_tsr + self.gamma * next_q * (1.0)
        self.critic.zero_grad()
        self.cnn.zero_grad()

        # print(f"state batch: {state_batch.shape}\n"
        #       f"action_batch: {action_batch.shape}")
        q_batch = self.critic([state_batch, action_batch])

        # print(f"q_batch: {q_batch.shape}")
        # print(f"tqbatch: {target_q_batch.shape}")
        critic_loss = nn.MSELoss()(q_batch, target_q_batch)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.cnn_optimizer.step()


        self.actor.zero_grad()
        policy_loss = self.critic([
            state_batch,
            self.actor(state_batch)
        ])
        policy_loss = policy_loss.mean()
        # policy_loss.backward()
        self.actor_optimizer.step()

        self._update_target(self.critic_target, self.critic)
        self._update_target(self.actor_target, self.actor)

        # update critic and actor networks

    def _update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def explore_action_val(self, state):
        """
        Choose an action for exploring the env
        """
        state_tsr = to_tensor_var(state)
        action_val = self.action(state_tsr)
        current_eps = (self.eps_start - self.eps_end) + self.eps_end
        noise = np.random.random() * current_eps
        action_val += noise
        return action_val

    def action(self, state):
        state = standardize(state)
        cstate = state.unsqueeze(0).unsqueeze(0)
        cstate = self.cnn(cstate)
        self.actor.eval()
        action_var = self.actor(cstate)
        if self.use_cuda:
            action = action_var.data.cpu().numpy()[0]
        else:
            action = action_var.data.numpy()[0]
        return action

    def best_action(self, state):
        state_tsr = state
        action_tsr = self.actor(state)
        return action_tsr

def to_tensor_var(x, use_cuda=False, dtype="float"):
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))


def gaussian_noise(shape=None, mu=.0, sigma=.1):
    noise = np.random.normal(loc=mu, scale=sigma, size=shape)
    return noise


def standardize(data):
    means = data.mean(dim=0, keepdim=True)
    stds = data.std(dim=0, keepdim=True)
    normalized_data = (data - means) / stds
    return normalized_data


def scale(data):
    maxv, minv = data.max(), data.min()
    scaled = (data - minv) / (maxv - minv)
    return scaled

if __name__ == "__main__":
    num_states = 3
    hidden_size = 100
    action_dim = 1
    # actor = ActorNet(num_states, hidden_size, num_actions)