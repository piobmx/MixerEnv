import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
device = torch.device('cpu')

class ActorCritic(nn.Module):
    def __init__(self, action_std_init=0.6, state_dim=3, action_dim=2, continuous_action=True,
                 hidden_size_1=64, hidden_size_2=64,):
        super(ActorCritic, self).__init__()
        hidden_size_1 = hidden_size_1
        hidden_size_2 = hidden_size_2
        self.has_continuous_action_space = continuous_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        if self.has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        if self.has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size_1),
                nn.Tanh(),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.Tanh(),
                nn.Linear(hidden_size_2, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size_1),
                nn.Tanh(),
                nn.Linear(hidden_size_1, hidden_size_2),
                nn.Tanh(),
                nn.Linear(in_features=hidden_size_2, out_features=action_dim),
                nn.Softmax(dim=-1)
            )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            return

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
