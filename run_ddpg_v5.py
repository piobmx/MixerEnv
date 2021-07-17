import os, time, pickle, json
from copy import deepcopy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from networks.Agent import DDPG
from networks.Evaluator import Evaluator
from mixer_agent import Mixer_agent
from utilz import *
from tqdm import tqdm

###
# Implementation of the beat-matching algorithm based on Deep Deterministic Policy Gradient
###

EPOCHS = 4000

ddpg_args = {}

evltr = Evaluator(EPOCHS)
ma = Mixer_agent(tracklist=testqueue, queue=testqueue)
ma.load_playlist()
ma.reset()
ma.reset_onsets()
# ma.nn_size = 50
input_dim, action_dim = ma.reset_pitch_by_frame()
print(action_dim, input_dim)
actor_lr = 0.001
critic_lr = 0.01
gamma = 0.85
target_tau = 0.001
hidden_size_1 = 32
hidden_size_2 = 32
ddpg_agent = DDPG(mixer_env=None, action_dim=action_dim, actor_lr=actor_lr, critic_lr=critic_lr, optimizer="sgd",
                  gamma=gamma, batch_size=50, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2,
                  target_tau=target_tau, models_size="mini", actor_input_size=input_dim, critic_input_size=input_dim,
                  norms=False)
# ma.random_playlist(5)
ddpg_agent.set_mixer(ma)
ddpg_agent.set_evaluator(evltr)
print(ma.playlist)
ma.set_rl_agent(ddpg_agent)
# ma.generate_original_overlays(saveto=True)
# ma.reset_pitch()
# ma.C = 0
# rp, rv, rp2, rv2 = ma.update_state()

epoch_rewards = []

# The main training loop starts:
now = nowness()
eval_dir = f"evaluation/{now}/"
os.makedirs(eval_dir, exist_ok=True)

write_agent_info(ddpg_agent, path=eval_dir)

max_episode_step = 400
# eps_decay_episodes = int(EPOCHS * 2 // 3)
eps_start, eps_end = 0.6, 0.03
eps_decay_episodes = 20
de_eps = (eps_start - eps_end) / eps_decay_episodes
eps = eps_start
total_reward = []
total_losses = []
total_action = []
show_elapse = False

for e in range(EPOCHS):
    _epoch_start_time = time.time()
    print(f"EPOCH: {e}. Memory size: {len(ddpg_agent.memory)}, eps: {eps}")
    ma.C = 0
    observation_pre = ma.default_state
    obs_size = observation_pre.size
    observation_post = None
    epoch_reward, epoch_losses = 0, np.array([0.0, 0.0])
    # de_eps += 0.005
    ma.audios[1].actionsum = np.zeros((1, ddpg_agent.action_dim))
    total_action.append([])
    for episode_step in tqdm(range(1, 180)):
        ma.episode = episode_step
        # if episode_step < eps_decay_episodes//2:
        if False:
            # action = ma.sample_beats_action_space(lims=[-.1, .1], action_num=ddpg_agent.action_dim)/20
            action = ma.sample_beats_action_space(lims=[-1, 1], action_num=ddpg_agent.action_dim)
        else:
            action = ddpg_agent.pitch_action(observation_pre, max(eps, 0))
        # observation_post, reward, done = ma.pitch_match_v2(action)
        action = action.reshape((action_dim, ))
        observation_post, reward, done, distance = ma.live_pitch_match(action, use_sox=True, update_live=False)
        ma.update_C()
        action = action.reshape((1, action_dim))
        post_obs_copy = deepcopy(observation_post)

        pre_obs = observation_pre.reshape((1, obs_size))
        next_obs = post_obs_copy.reshape((1, obs_size))
        _action = action
        reward /= 10
        ma.audios[1].actionsum += action
        ddpg_agent.observe(pre_obs, [_action], next_obs, [[reward]], verbose=False)
        [critic_loss, policy_loss] = ddpg_agent.train(verbose=False)
        epoch_losses += np.array([critic_loss, policy_loss])

        # observation_pre = deepcopy(observation_post)
        observation_pre = ma.get_next_pre_state()
        ddpg_agent.current_losses = [0, 0]
        action_value = action
        ddpg_agent.n_episode += 1
        ddpg_agent.evaluator.losses_history[e].append([critic_loss, policy_loss])
        ddpg_agent.evaluator.reward_history[e].append(reward)
        ddpg_agent.evaluator.action_history[e].append(action_value)
        epoch_reward += reward
        if done:
            print(f"DONE after {episode_step} steps")
            break
        total_action[-1].append(action.item())
    total_reward.append(epoch_reward)
    total_losses.append(epoch_losses)
    # ma.audios[1].generate_waveform_from_sox_log()
    print("losses:", total_losses[-1], "reward:", epoch_reward)
    print("accumulated action value: ", ma.audios[1].actionsum)
    if np.random.random() < 0.01:
        ma.plot_pitches(eval_dir, step=e)

    ddpg_agent.evaluator.update_epoch()

    ddpg_agent.current_losses = [0, 0]
    ddpg_agent.evaluator.save_reward_tofile(e, dir=eval_dir)
    np.save(eval_dir + "total_action.npy", total_action)

    # if np.random.random() < 0.001:
    #     ddpg_agent.mixer.generate_new_overlays(wavname=f"ddpgnew{e}", saveto=True)
    # ddpg_agent.mixer.reset(n=None)
    if e % 12 == 1 or e < 12:
        ddpg_agent.save_checkpoint(name="checkpoint", last_timestep=e, dir=eval_dir)
        fig, ax = plt.subplots(3, 1, figsize=(20, 10), dpi=120)
        ax[0].plot(total_reward)
        ax[0].set_title("REWARD OVER EPOCHS")
        plot_losses = np.array(total_losses.copy()).T
        ax[1].plot(plot_losses[0])
        ax[1].set_title("CRITIC LOSS")
        ax[2].plot(plot_losses[1])
        ax[2].set_title("POLICY LOSS")
        plt.savefig(eval_dir+"mid.png")
        plt.close()
    _epoch_elapse_time = time.time() - _epoch_start_time
    print(f"epoch {e} elapses for: {_epoch_elapse_time}")
    ddpg_agent.evaluator.update_epoch_elapse_time(e, _epoch_elapse_time)
    if e % 12 == 0:
        print(f"RESET ENVIRONMENT AT EPOCH {e}")
        # ma.episodic_reset_pitch_by_frame()
        eps -= de_eps
    ma.episodic_reset_pitch_by_frame()