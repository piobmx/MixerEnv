import argparse
import os
import time
from copy import deepcopy

import librosa.display
import matplotlib
matplotlib.use('Qt5Agg')
import sox
from matplotlib import pyplot as plt
from networks.Agent import DDPG
from networks.PPO import PPO
from networks.Evaluator import Evaluator
from mixer_agent import Mixer_agent
from Actions import Actions
from utilz import *
from tqdm import tqdm
from scipy.io.wavfile import write


def construct_mixer(tracks):
    mixer = Mixer_agent(tracklist=tracks, queue=tracks)
    mixer.load_playlist()
    mixer.reset()
    mixer.reset_onsets()
    return mixer


def construct_models(config, mixer):
    ddpg_agent = DDPG(mixer_env=mixer, action_dim=config.ddpg_action_dim, actor_lr=config.ddpg_actor_lr,
                      critic_lr=config.ddpg_critic_lr, optimizer=config.ddpg_optimizer,
                      gamma=config.ddpg_gamma, batch_size=config.ddpg_batch_size, target_tau=config.ddpg_target_tau,
                      actor_input_size=mixer.nn_size, critic_input_size=mixer.nn_size,
                      hidden_size_1=config.ddpg_hidden_size_1, hidden_size_2=config.ddpg_hidden_size_2,
                      norms=config.norms)
    ppo_agent = PPO(state_dim=config.ppo_state_dim, action_dim=config.ppo_action_dim,
                    actor_lr=config.ppo_actor_lr, critic_lr=config.ppo_critic_lr,
                    gamma=config.ppo_gamma, K_epochs=config.k_epochs, eps_clip=config.eps_clip,
                    has_continuous_action_space=config.ppo_has_continuous_action_space,
                    action_std_init=config.action_std,
                    hidden_size_1=config.ppo_hidden_size_1, hidden_size_2=config.ppo_hidden_size_2)

    return ddpg_agent, ppo_agent

def train_1(rl_agent, mixer, epochs=20):
    EPOCHS = epochs
    evltr = Evaluator(EPOCHS)
    rl_agent.set_mixer(mixer)
    rl_agent.set_evaluator(evltr)
    mixer.set_rl_agent(rl_agent)
    write_agent_info(rl_agent, path=eval_dir)

    max_episode_step = 400
    eps_start, eps_end = 0.4, 0.03
    eps_decay_episodes = 30
    de_eps = (eps_start - eps_end) / eps_decay_episodes
    eps = eps_start
    epoch_rewards = []
    total_reward = []
    total_losses = []
    show_elapse = False
    trackA, trackB = None, None
    best_match = 0
    for e in range(EPOCHS):
        _epoch_start_time = time.time()
        print(f"EPOCH: {e}")
        observation_pre = mixer.get_beats_state(track_index=1)
        print(observation_pre.size)
        obs_size = observation_pre.size
        observation_post = None
        epoch_reward = 0
        epoch_losses = np.array([0.0, 0.0])
        # eps = eps_start
        mixer.audios[1].actionsum = np.zeros(rl_agent.action_dim, )
        for episode_step in tqdm(range(0, max_episode_step)):
            mixer.episode = episode_step
            if episode_step < 1:
                action = mixer.sample_beats_action_space(lims=[-1, 1], action_num=rl_agent.action_dim)
            else:
                action = rl_agent.padding_action(observation_pre, max(eps, 0))
            observation_post, reward, done = mixer.beat_match(action, step=2)
            post_obs_copy = deepcopy(observation_post)

            pre_obs = observation_pre.reshape((1, obs_size))
            next_obs = post_obs_copy.reshape((1, obs_size))
            _action = action.reshape((action.size, ))
            mixer.audios[1].actionsum += action.reshape((action.size, ))
            rl_agent.observe(pre_obs, [[_action]], next_obs, [[reward/10]], verbose=False)
            [critic_loss, policy_loss] = rl_agent.train(verbose=False)
            epoch_losses += np.array([critic_loss, policy_loss])

            observation_pre = deepcopy(observation_post)
            rl_agent.current_losses = [0, 0]
            action_value = action
            rl_agent.n_episode += 1
            rl_agent.evaluator.losses_history[e].append([critic_loss, policy_loss])
            rl_agent.evaluator.reward_history[e].append(reward)
            if action_value.size > 0:
                rl_agent.evaluator.action_history[e].append(action_value)
            else:
                rl_agent.evaluator.action_history[e].append(None)
            epoch_reward += reward
            if done:
                print(f"EPISODE {e} is done after {episode_step} steps")
                break
        eps -= de_eps
        if mixer.best_num_in_sync > 4:
            print(f"Generating tracks...")
            trackA, trackB = mixer.audios[0].y.copy(), mixer.audios[1].Y_1.copy()
            padding_b = mixer.audios[1].start_location
            print(f"START LOCATION: {padding_b}")
            mixer.plot_newest_beats(eval_dir, step="main")
            break
        if mixer.best_num_in_sync > best_match:
            best_match = mixer.best_num_in_sync
            padding_b = mixer.audios[1].start_location
            trackA, trackB = mixer.audios[0].y.copy(), mixer.audios[1].Y_1.copy()
        mixer.audios[1].generate_waveform_from_sox_log()
        # if True:
        #     mixer.plot_newest_beats(eval_dir, step=e)
        rl_agent.evaluator.update_epoch()
        rl_agent.current_losses = [0, 0]
        rl_agent.evaluator.save_reward_tofile(e, dir=eval_dir)
        mixer.episodic_onset_reset()
        _epoch_elapse_time = time.time() - _epoch_start_time
        print(f"epoch {e} elapses for: {_epoch_elapse_time}")
        rl_agent.evaluator.update_epoch_elapse_time(e, _epoch_elapse_time)
    print("DDPG Done...")
    return [trackA, trackB, padding_b]

def train_2(mixer, agent):
    len1 = len(mixer.audios[0].audio_frames)
    print("START pitch matching")
    new_states = []
    new_track = []
    rewards = []
    for i in range(len1):
        if min(librosa.amplitude_to_db(mixer.audios[1].audio_frames[i])) > -98 and\
                min(librosa.amplitude_to_db(mixer.audios[0].audio_frames[i])) > -98:
            f1 = mixer.CHROMA_A[i]
            f2 = mixer.CHROMA_B[i]
            pr = l2_reward(f1, f2)
            distance = (f1 - f2) / 12
            distance = distance.reshape((1, 12))
            action = agent.select_action(distance)
            # action = [np.random.random() * 6]
            pitchsox = sox.Transformer();
            norm_level = 6 * action[0]
            # print(norm_level)
            pitchsox.pitch(norm_level, quick=True)
            modulated = soxx(pitchsox, mixer.audios[0].audio_frames[i])
            new_state = librosa.feature.chroma_stft(modulated,
                                                    sr=global_sr,
                                                    hop_length=4410,
                                                    center=False)
            mixer.audios[0].audio_frames[i] = modulated
            new_track.append(modulated)
            new_states.append(new_state)
            pa = l2_reward(new_state, f2)
            rewards.append(pa)
        else:
            new_states.append(np.zeros((12, 1)))
            new_track.append(mixer.audios[0].audio_frames[i])
    # print(mixer.audios[0].audio_frames)
    print(np.array(new_track).shape)
    track1 = join_frames(new_track)
    print(f"TRACK!: {(new_track)}")
    rate = 0
    for r in rewards:
        if r == 1.0:
            rate += 1
    print(rate)

    mixer.plot_pitches(eval_dir=eval_dir, step="mainpitch", to_plot=np.array(new_states))
    return track1

def fade_two_tracks(track1, track2, fade_len, pad_len):
    fade_shape_1, fade_shape_2 = "h", "q"
    fad1 = sox.Transformer()
    fad1.fade(fade_out_len=fade_len, fade_shape=fade_shape_1)

    fad2 = sox.Transformer()
    fad2.fade(fade_in_len=fade_len/2, fade_out_len=1.5, fade_shape=fade_shape_2)
    fad2.pad(start_duration=pad_len)

    path1, path2 = f"{eval_dir}tmp1.wav", f"{eval_dir}tmp2.wav"
    pathx = f"{eval_dir}x.wav"
    T1 = fad1.build(input_array=track1.copy(), sample_rate_in=global_sr,
                    output_filepath=path1)
    T2 = fad2.build(input_array=track2.copy(), sample_rate_in=global_sr,
                    output_filepath=path2)
    # mixer.plot_pitches(eval_dir=eval_dir, step="mainpitch", to_plot=np.array(new_states))
    cbn = sox.combine.Combiner()
    cbn.build(
        input_filepath_list=[path1, path2],
        output_filepath=pathx,
        combine_type="mix"
    )
    print("#")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpg_action_dim", default=2, type=int)
    parser.add_argument("--ddpg_actor_lr", default=.0001, type=float)
    parser.add_argument("--ddpg_critic_lr", default=.0001, type=float)
    parser.add_argument("--ddpg_optimizer", default="adam", type=str)
    parser.add_argument("--ddpg_gamma", default=0.89, type=float)
    parser.add_argument("--ddpg_batch_size", default=50, type=int)
    parser.add_argument("--ddpg_target_tau", default=0.01, type=float)
    parser.add_argument("--ddpg_hidden_size_1", default=64, type=int)
    parser.add_argument("--ddpg_hidden_size_2", default=64, type=int)
    parser.add_argument("--norms", default=True, type=bool)

    parser.add_argument("--k_epochs", default=80, type=int)
    parser.add_argument("--ppo_hidden_size_1", default=64, type=int)
    parser.add_argument("--ppo_hidden_size_2", default=64, type=int)
    parser.add_argument("--eps_clip", default=0.2, type=float)
    parser.add_argument("--ppo_gamma", default=0.8, type=float)
    parser.add_argument("--ppo_actor_lr", default=.001, type=float)
    parser.add_argument("--ppo_critic_lr", default=.01, type=float)
    parser.add_argument("--action_std", default=0.05, type=float)
    parser.add_argument("--ppo_state_dim", default=12, type=int)
    parser.add_argument("--ppo_action_dim", default=1, type=int)
    parser.add_argument("--ppo_has_continuous_action_space", default=True, type=bool)

    queue = [
        "/Users/wxxxxxi/Projects/ReinL/test_folder/for_main/Inti.wav",
        "/Users/wxxxxxi/Projects/ReinL/test_folder/for_main/Kappelberg Chant.wav",
    ]
    # EPOCHS = 2000

    config = parser.parse_args()

    mixer = construct_mixer(tracks=queue)
    ddpg_model, ppo_model = construct_models(config=config, mixer=mixer)
    now = nowness()
    eval_dir = f"evaluation/{now}/"
    os.makedirs(eval_dir, exist_ok=True)

    pth = "/Users/wxxxxxi/Projects/ReinL/evaluation/630_h5m48s33/1305000_PPO_preTrainedPPO_Pendulum-v0_0_0.pth"
    ppo_model.load(pth)

    fade_out_1 = mixer.audios[0].beats_in_seconds[-1] - mixer.audios[1].BEATS[1]
    fade_in_2 = fade_out_1

    track1, track2, pd = train_1(ddpg_model, mixer, epochs=10)
    track_file_1 = eval_dir + f"beatmatchedA.wav"
    track_file_2 = eval_dir + f"beatmatchedB.wav"
    write(track_file_1, 44100, track1)
    write(track_file_2, 44100, track2)
    pad = sox.Transformer()
    pad.pad(start_duration=(pd))
    t2 = soxx(pad, track2)
    track_file_2 = eval_dir + f"beatmatchedB_padded.wav"
    write(track_file_2, 44100, t2)

    mixer = construct_mixer(tracks=[track_file_1, track_file_2])
    mixer.reset_pitch_by_frame()
    modulated_track1 = train_2(mixer, ppo_model)
    print(mixer.audios[0].beats_in_seconds)
    print(mixer.audios[1].BEATS)
    print(f"{mixer.audios[0].beats_in_seconds[-1]} - {mixer.audios[1].BEATS[1]}")
    fade_two_tracks(modulated_track1, track2, fade_len=fade_out_1, pad_len=pd)
    # train(tracklist=testqueue)