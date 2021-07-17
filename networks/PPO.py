from networks.ActorCritic import *
import torch
import torch.nn as nn

class PPOBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logp = []
        self.rewards = []

    def clear(self):
        del self.actions[:], self.states[:], self.logp[:], self.rewards[:]


class PPO:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, hidden_size_1, hidden_size_2, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = PPOBuffer()
        self.policy_net = ActorCritic(state_dim=state_dim, action_dim=action_dim,
                                      action_std_init=action_std_init, hidden_size_1=hidden_size_1,
                                      hidden_size_2=hidden_size_2,
                                      continuous_action=has_continuous_action_space).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.actor.parameters(), 'lr': actor_lr},
            {'params': self.policy_net.critic.parameters(), 'lr': critic_lr}
        ])
        self.policy_old = ActorCritic(state_dim=state_dim, action_dim=action_dim,
                                      action_std_init=action_std_init, hidden_size_1=hidden_size_1,
                                      hidden_size_2=hidden_size_2,
                                      continuous_action=has_continuous_action_space,
                                      ).to(device)

        self.policy_old.load_state_dict(self.policy_net.state_dict())
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        """
        Set original standard deviation for noise generation
        """
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy_net.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            pass

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """
        Decay standard deviation value for noise generation as the training progresses
        """
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                pass
            self.set_action_std(self.action_std)
        else:
            pass
        
    def select_action(self, state):
        """
        Selection next action, based on the input state
        """
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            self.buffer.states.append(state.reshape((1, state.nelement())))
            self.buffer.actions.append(action)
            self.buffer.logp.append(action_logprob)
            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logp.append(action_logprob)
            return action.item()

    def update(self):
        """
        Main part of the algorithm
        """
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalization
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logp, dim=0)).detach().to(device)

        t_loss = 0
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_net.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            # Calculating the r(\theta)
            r = torch.exp(logprobs - old_logprobs.detach())
            # Surrogate Loss
            advantade_function = rewards - state_values.detach()
            surr1 = r * advantade_function
            surr2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantade_function
            # loss of PPO objective
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) \
                   - 0.01 * dist_entropy
            # take a gradient step
            t_loss += loss.mean().detach().item()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # print(t_loss)
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_net.state_dict())
        self.buffer.clear()
        return t_loss

    def save(self, to_path):
        torch.save(self.policy_old.state_dict(), to_path)

    def load(self, from_path):
        self.policy_old.load_state_dict(torch.load(from_path, map_location=lambda storage, loc: storage))
        self.policy_net.load_state_dict(torch.load(from_path, map_location=lambda storage, loc: storage))

