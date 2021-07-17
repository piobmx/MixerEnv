# from actor import *
import pickle, os, logging, gc

from networks.critic import critic_mini, MelCritic
from networks.actor import actor_mini, MelActor
from networks.Memory import ReplayMemory, Transition
from networks.Evaluator import Evaluator
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class DDPG:
    """
    Implimentation of Deep Deterministic Policy Gradient algorithm following this paper: https://arxiv.org/pdf/1509.02971.pdf
    Consisting of a Actor and a Critic and a Replay Buffer of size 20000.
    """

    def __init__(self, mixer_env, state_dim=0, action_dim=1, memory_size=60000, actor_lr=0.01,
                 critic_lr=0.01, optimizer="sgd", gamma=0.9, episode_to_min_eps=200,
                 batch_size=50, eps_start=0.95, eps_end=0.01, target_tau=0.05, use_cuda=False, models_size="mini",
                 critic_input_size=13, actor_input_size=12, norms=False, hidden_size_1=128, hidden_size_2=128):
        self.mixer = mixer_env
        self.gamma, self.eps_start, self.eps_end = gamma, eps_start, eps_end
        self.episode_to_min_eps = episode_to_min_eps
        self.de_eps = (self.eps_start - self.eps_end) / self.episode_to_min_eps
        self.target_tau = target_tau
        self.n_episode = 0
        self.use_cuda = use_cuda
        self.evaluator = None

        # self.cnn = AudioCNN()
        # self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=1e-4)
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.a_hidden_size = action_hidden_size
        # self.c_hidden_size = critic_hidden_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.default_loss = self.current_losses = [0, 0]

        self.memory = ReplayMemory(capacity=memory_size)
        # self.actor = Actor(nb_status=30, nb_actions=1)
        # self.critic = Critic(nb_status=30, nb_actions=1)
        if models_size == "full":
            self.actor = MelActor()
            self.critic = MelCritic()
            self.actor_target = deepcopy(self.actor)
            self.critic_target = deepcopy(self.critic)
        if models_size == "mini":
            self.actor = actor_mini(input_size=actor_input_size, output_size=self.action_dim,
                                    action_dim=self.action_dim, norm=norms,
                                    hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2)
            self.critic = critic_mini(input_size=critic_input_size, action_dim=self.action_dim,
                                      output_size=self.action_dim, norm=norms,
                                      hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2)
            self.actor_target = deepcopy(self.actor)
            self.critic_target = deepcopy(self.critic)

        self.criterion = nn.MSELoss()

        if optimizer == "adam":
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        else:
            self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.001)
            self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.001)

        self.checkpoint_dir = "OUTPUTS/checkpoints_dir/"
        self.graph_dir = "OUTPUTS/graph/"
        self.outputs_dir = "OUTPUTS/audios/"
        self.log_dir = "OUTPUTS/evaluation/"

    def set_mixer(self, mixer):
        self.mixer = mixer

    def update_episode(self):
        self.n_episode += 1
        return

    def set_discount(self, discount):
        self.gamma = discount
        return

    def set_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def interact(self, epoch):
        """
        interaction between rl agent and mixer.
        """
        # reward, mdval = self.mixer.update_one_step()

        self.evaluator.reward_history[epoch].append(reward)
        self.evaluator.losses_history[epoch].append(self.current_losses)
        self.evaluator.action_history[epoch].append(mdval)
        self.current_losses = [0, 0]

    def log_history(self, epoch):
        """
        interaction between rl agent and mixer.
        """
        # reward, mdval = self.mixer.update_one_step()

        self.evaluator.reward_history[epoch].append(reward)
        self.evaluator.losses_history[epoch].append(self.current_losses)
        self.evaluator.action_history[epoch].append(mdval)
        self.current_losses = [0, 0]

    def train(self, verbose=False):
        """
        Where main training is happening. Policies are updated via training Critic and Actor networks.
        """
        if len(self.memory) < self.batch_size:
            # if size of ReplayBuffer is smaller than batch size then skip this step
            return 0, 0

        # Sample a minibatch of (state, action, next_state, reward) from ReplayBuffer and process the data.
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_tsr = to_tensor_var(batch.state)
        action_batch = to_tensor_var(batch.action)
        reward_batch = to_tensor_var(batch.reward)
        next_state_tsr = to_tensor_var(batch.next_state)

        if verbose:
            print(f"state_tsr: {state_tsr.shape}\n"
                  f"action_tsr: {action_batch.shape} batch: {action_batch.shape}\n"
                  f"reward_tsr: {action_batch.shape} batch: {reward_batch.shape}\n"
                  f"next_state_tsr:{next_state_tsr.shape}")

        # next_state_tsr = standardize(next_state_tsr)
        # state_tsr = standardize(state_tsr)

        # Update critic by minimizing the MSE Loss

        # next_action_batch = self.actor_target(next_state_tsr).unsqueeze(1)  # mixer
        next_action_batch = self.actor_target(next_state_tsr)  # gym

        # assert next_action_batch.shape == (self.batch_size, 1, 1)
        frame_n = 2205
        # assert next_state_tsr.shape == (self.batch_size, 2, frame_n)
        # print(next_state_tsr.shape, next_action_batch.shape)
        future_q_vals = self.critic_target([next_state_tsr, next_action_batch])

        # self.set_discount(0.99)
        # print(f"gamma: {self.gamma} \t rb {reward_batch} \n"
        #       f"future_q_vals {future_q_vals}")
        target_q_batch = (reward_batch) + self.gamma * future_q_vals

        q_batch = self.critic([state_tsr, action_batch])
        # print(f"qb: {q_batch.shape}, tqb: {target_q_batch.shape}")
        flat_qb, flat_tqb = q_batch.view(-1), target_q_batch.view(-1)
        # assert flat_qb.shape == flat_tqb.shape
        critic_loss = self.criterion(flat_qb, flat_tqb)
        # print(f"critic loss: {critic_loss.detach().numpy()}")
        cls = critic_loss.sum()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor using policy gradient.
        # self.actor.zero_grad()

        to_loss_actions = self.actor(state_tsr)
        policy_loss = -self.critic([
            state_tsr,
            to_loss_actions,
        ])
        # print(f"policy loss: {policy_loss.detach().numpy()}")
        policy_loss = policy_loss.mean()
        # print(f"cls: {cls}, policy loss: {policy_loss}")
        self.current_losses = [cls, policy_loss]
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        # Soft updating both the target networks
        self._update_target(self.critic_target, self.critic)
        self._update_target(self.actor_target, self.actor)
        return cls.item(), policy_loss.item()

    def _update_target(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau
            )

    def explore_action_val(self, state, eps):
        """
        Choose an action for exploring the env, the extend of exploring depends of the epsilon value (exploration / exploitation trade-off)
        The action value is received by the mixer and corresponding action is subsequently executed.
        """
        state_tsr = to_tensor_var(state)
        # state_tsr = state_tsr.unsqueeze(0).unsqueeze(0)
        action_val = self.action(state_tsr.detach())
        # print(f"action_val: {action_val}")
        # current_eps = max((self.eps_start - self.eps_end) - self.de_eps, 0.01)
        # noise = ((-1) + 2 * np.random.random()) * current_eps
        if eps == -1:
            return action_val
        action_val = action_val + np.random.normal(0, eps / 2, size=1)
        return action_val

    def action(self, state):
        # state = standardize(state)
        if state.dim() != 4:
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

    def pitch_action(self, state, eps):
        state = state.reshape((1, state.size))
        state_tsr = to_tensor_var(state)
        action = self.actor(state_tsr)
        noise = np.random.normal(0, eps, size=self.action_dim)
        action = (action.detach().numpy() + noise).squeeze(1)
        action = action.clip(-1, 1)
        # print(f"ACTION: {action}")
        return action

    def padding_action(self, state, eps):
        state = state.reshape((1, state.size))
        state_tsr = to_tensor_var(state)
        action = self.actor(state_tsr) / 20

        if action.nelement() > 1:
            action = action.view(-1).detach().numpy()
            noise = np.random.normal(0, eps / 10, size=action.shape)
        else:
            action = action.item()
            noise = np.random.normal(0, eps / 10, size=1)
        action = action + noise
        action = action.clip(-1/20, 1/20)

        return action

    def best_action_gym(self, state, eps):
        state = state.reshape((1, 3))
        state_tsr = to_tensor_var(state)
        action = self.actor(state_tsr) * 2
        action = action.item()
        action = action + np.random.normal(0, eps / 2, size=1)
        action = action.clip(-2., 2.)

        return action

    def observe(self, state, action, next_state, reward, verbose=False):
        if verbose:
            print(f"ostate: {state}\n"
                  f"oaction: {action}\n"
                  f"onstate: {next_state}\n"
                  f"oreward: {reward}")
        self.memory.push(state, action[0], next_state, reward)
        return

    def save_checkpoint(self, last_timestep, name, dir=None):
        """
        Saving network parameters to a file
        Arguments:
            last_timestep:  Last timestep in training before saving
        """
        if dir is not None:
            checkpoint_dir = self.checkpoint_dir + f'{name}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = checkpoint_dir + f'/ep_{last_timestep}.pth'
        else:
            checkpoint_dir = dir + f"{name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = checkpoint_dir + f'/ep_{last_timestep}.pth'

        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'actor_lr': self.actor_lr,
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic_lr': self.critic_lr,
            'replay_buffer': self.memory,
        }
        torch.save(checkpoint, checkpoint_name)
        return

    def load_checkpoint(self, checkpoint_path=None):
        """
        """

        if os.path.isfile(checkpoint_path):
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.memory = checkpoint['replay_buffer']

            return start_timestep
        else:
            raise OSError('Checkpoint not found')


def to_tensor_var(x, use_cuda=False, dtype="float"):
    """
    Turn the input "x" into trainable tensor.
    """
    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    long_tensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    byte_tensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(float_tensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(long_tensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(byte_tensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(float_tensor(x))
