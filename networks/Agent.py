# from actor import *
import pickle

from networks.critic import Critic
from networks.actor import Actor
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
    """
    Implimentation of Deep Deterministic Policy Gradient algorithm following this paper: https://arxiv.org/pdf/1509.02971.pdf
    Consisting of a Actor and a Critic and a Replay Buffer of size 20000.
    """
    def __init__(self, mixerEnv, state_dim=0, action_dim=1, memory_size=20000, actor_output_act=F.tanh, actor_lr=0.01,
                 critic_lr=0.001, optimizer="sgd", gamma=0.99, action_hidden_size=32, critic_hidden_size=32,
                 batch_size=50, eps_start=0.95, eps_end=0.01, target_tau=0.001, use_cuda=False):
        self.mixer = mixerEnv
        self.gamma, self.eps_start, self.eps_end = gamma, eps_start, eps_end
        self.target_tau = target_tau
        self.n_episode = 0
        self.use_cuda = use_cuda
        self.evaluator = None

        # self.cnn = AudioCNN()
        # self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=1e-4)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_hidden_size = action_hidden_size
        self.c_hidden_size = critic_hidden_size
        self.a_output_act = actor_output_act
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.current_losses = [0, 0]

        self.memory = ReplayMemory(capacity=memory_size)
        # self.actor = Actor(nb_status=30, nb_actions=1)
        # self.critic = Critic(nb_status=30, nb_actions=1)
        self.actor = Actor()
        self.critic = Critic()
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        if optimizer == "adam":
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        else:
            self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.1)
            self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.1)

    def update_episode(self):
        self.n_episode += 1
        return

    def set_discount(self, discount):
        self.discount = discount
        return

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def interact(self, epoch):
        reward = self.mixer.update_one_step()
        self.evaluator.reward_history[epoch].append(reward)
        self.evaluator.losses_history[epoch].append(self.current_losses)
        self.current_losses = [0, 0]


    def train(self):
        """
        Where main training is happening. Policies are updated via training Critic and Actor networks.

        """
        if (len(self.memory) < self.batch_size):
            # if size of ReplayBuffer is smaller than batch size then skip this step
            return

        # Sample a minibatch of (state, action, next_state, reward) from ReplayBuffer and process the data.
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_tsr = to_tensor_var(batch.state)
        action_tsr = to_tensor_var(batch.action)
        action_batch = action_tsr.unsqueeze(1).unsqueeze(1)

        reward_tsr = to_tensor_var(batch.reward)
        reward_batch = reward_tsr.unsqueeze(1).unsqueeze(1)
        next_state_tsr = to_tensor_var(batch.next_state)
        # print(f"state_tsr: {state_tsr.shape}\n"
        #       f"action_tsr: {action_batch.shape}\n"
        #       f"reward_tsr: {reward_batch.shape}\n"
        #       f"next_state_tsr:{next_state_tsr.shape}")

        # next_state_tsr = standardize(next_state_tsr)
        # state_tsr = standardize(state_tsr)


        # Update critic by minimizing the MSE Loss
        next_action_batch = self.actor_target(next_state_tsr).unsqueeze(1)
        assert next_action_batch.shape == (self.batch_size, 1, 1)
        assert next_state_tsr.shape == (self.batch_size, 2, 4410)
        future_q_vals = self.critic_target([next_state_tsr, next_action_batch])
        # future_q_vals.volatile = False
        self.set_discount(0.85)
        # print(f"rb {reward_batch.shape}")
        target_q_batch = to_tensor_var(reward_batch) + \
                         self.discount * future_q_vals
        self.critic.zero_grad()
        # action_batch = action_batch.reshape([50, 1, 1])
        assert action_batch.shape == (self.batch_size, 1, 1)
        assert state_tsr.shape == (self.batch_size, 2, 4410)
        q_batch = self.critic([state_tsr, action_batch])
        # print(f"qb: {q_batch.shape}, tqb: {target_q_batch.shape}")
        flat_qb, flat_tqb = q_batch.view(-1), target_q_batch.view(-1)
        assert flat_qb.shape == flat_tqb.shape
        critic_loss = nn.MSELoss()(flat_qb, flat_tqb)
        cls = critic_loss.sum()
        # print(f"cls: {cls}")
        critic_loss.backward()
        self.critic_optimizer.step()


        # Update the actor using policy gradient.
        self.actor.zero_grad()

        # state_tsr.volatile = False
        to_loss_actions = self.actor(state_tsr).unsqueeze(1)

        #
        policy_loss = -self.critic([
            state_tsr,
            to_loss_actions,
            # self.actor(state_tsr).unsqueeze(1)
        ])
        policy_loss = policy_loss.mean()
        # print(f"policy loss shape: {policy_loss.shape} - mean; {policy_loss}")
        self.current_losses = [cls, policy_loss]
        policy_loss.backward()

        # self.actor_optimizer.step()

        # Soft updating both the target networks
        self._update_target(self.critic_target, self.critic)
        self._update_target(self.actor_target, self.actor)




    def _update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def explore_action_val(self, state):
        """
        Choose an action for exploring the env, the extend of exploring depends of the epsilon value (exploration / exploitation trade-off)
        The action value is received by the mixer and corresponding action is subsequently executed.
        """
        state_tsr = to_tensor_var(state)
        # state_tsr = state_tsr.unsqueeze(0).unsqueeze(0)
        action_val = self.action(state_tsr)
        # print(f"action_val: {action_val}")
        current_eps = (self.eps_start - self.eps_end) + self.eps_end
        noise = np.random.random() * current_eps / 20000

        action_val += noise
        return action_val

    def action(self, state):
        # state = standardize(state)
        if state.dim() != 3:
            state = state.unsqueeze(0)
        self.actor.eval()
        action_var = self.actor(state.detach())
        # print(f"actionvar: {action_var}, at{self.mixer.C}")
        # if True in torch.isnan(action_var) or abs(action_var.data.numpy()[0]) > 1:
        #     return np.array([0.0], dtype=np.float64)
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
    """
    Turn the input "x" into trainable tensor.
    """
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
    """
    """
    noise = np.random.normal(loc=mu, scale=sigma, size=shape)
    return noise


# def standardize(data):
#     """
#     Standardize input
#     """
#     means = data.mean(dim=0, keepdim=True)
#     stds = data.std(dim=0, keepdim=True)
#     normalized_data = (data - means) / stds
#     return normalized_data
#
# def scale(data):
#     """
#     Scale input
#     """
#     maxv, minv = data.max(), data.min()
#     scaled = (data - minv) / (maxv - minv)
#     return scaled