import random, math
from collections import namedtuple, Counter

from mixer_agent import Mixer_agent
from utilz import *
from Actions import Actions
from DQN import DQN, ReplayMemory
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


actions = Actions()

ma = Mixer_agent(tracklist=testqueue, actions=actions, queue=testqueue)
# ma.random_playlist(5)
ma.load_playlist()
print(ma.playlist)
ma.reset()
ma.generate_original_overlays(saveto=True)
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = 3
policy_net = DQN(4410, 3).to(device)
target_net = DQN(4410, 3).to(device)
target_net.eval()
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

steps_done = 0
def select_action(state, mixer):
    if not ma.both_playing:
        return 0
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


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
    # print(f"{state_batch.dtype} - {action_batch.dtype}")
    action_batch = action_batch.type(torch.LongTensor)
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

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

epochs = 120
action_counter = Counter()
epoch_rewards = []
for e in (range(epochs)):
    #     observation = ma.update_observation
    ma.C = 0
    timestep = 0
    print(f"EPOCH: {e}...")
    #     current
    rp, rv, rp2, rv2 = ma.update_state()
    rv = torch.Tensor(rv)
    epoch_reward = 0
    for i in tqdm(range(0, 500)):
    # while timestep < 445:
    #     if timestep % 120 == 0:
    #         print(f"timestep: {timestep}")
        timestep += 1
        # print(rv.shape)
        action = select_action(rv, mixer=ma)
        action_counter[str(action)] += 1
        # print(f"action: {action}")
        # ma.select_actions()
        if action != 0:
            actionID = action.item()
        else: actionID = 0
        ma.set_action(actionID)
        # print(f"actionid: {actionID}")
        ma.exec_action(ma.action_to_exec)
        ma.update_C()
        nrp, nrv, nrp2, nrv2 = ma.update_state()

        # ma.update_observation()
        ma.update_before_after_observation()
        #         nrp, nrv, nrp2, nrv2 = ma.update_state()

        aprime_reward = ma.reward()
        epoch_reward += aprime_reward
        aprime_reward = torch.Tensor([aprime_reward], ).to(device)
        actionID = torch.Tensor([[int(actionID)]], ).to(device)
        nrv = torch.Tensor(nrv)
        rv = torch.Tensor(rv)
        nrv = nrv.reshape((1, 1, nrv.size()[-1]))
        rv = rv.reshape((1, 1, rv.size()[-1]))

        # print(actionID)
        if ma.both_playing:
            memory.push(rv, actionID, nrv, aprime_reward)
            optimize_model()
        rv = nrv
    epoch_rewards.append(epoch_reward)
    # ma.save_current(e)
    if e < epochs - 1:
        ma.save_audios_history(e)
        ma.recover_audios()
    ma.save_amp_history(e)

#         aprime_reward = ma.calc_reward()
#         update_q(space, rem_pos, rem_vel, next_action, new_reward)
#         rem_pos, rem_vel = new_pos, new_vel


# In[646]:
print(epoch_rewards)

# ma.waveplot_compare()
# ov = ma.generate_original_overlays(saveto=True)
# write("real.wav", global_sr, ma.original_copy)
# ipd.Audio(ov, rate=global_sr)


# In[647]:

print(action_counter)
for a in ma.audios:
    print(len(a.frame_log))


# In[648]:
#
#
# for a in ma.audios:
#     print(len(a.audio_frames))
ma.generate_new_overlays("new", saveto=True)
ma.generate_original_overlays("ori", saveto=True)
