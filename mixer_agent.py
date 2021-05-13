#!/usr/bin/env python
# coding: utf-8

# In[597]:

from utilz import *
from Actions import Actions
from Audio import Audio
from Rewarder import Rewarder
from DQN import DQN, ReplayMemory
import datetime
import os
import sys
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pydub
import librosa
import librosa.display
import soundfile as sf
from pydub import AudioSegment
from scipy.io.wavfile import write

import IPython.display as ipd
import ipywidgets as widgets

print(sys.version)
print(sys.executable)

# In[685]:


FRAMES_PER_STEP = 10


class Mixer_agent():
    """
    Main mixer environment.

    """

    def __init__(self, tracklist, actions, queue=[]):
        self.tracklist = tracklist
        self.track_num = 5
        self.queue = queue
        self.audios = []
        self.rl_agent = None

        self.original = None

        self.step = 0
        self.episode = 0
        self.actions = actions
        self.actions.load_action_dict()
        self.action_history = []
        self.DQN = DQN(4410, len(self.actions))  # TODO: Neural Network Architectures

        self.active_tracks = None
        self.current_track = None
        self.next_track = None
        self.observation = None
        self.current_obs = []
        self.rewarder = Rewarder()
        self.next_action = None
        self.both_playing = False
        self.action_to_exec = self.actions.no_action

        self.C = 0

    def reset(self):
        #         reset q table
        #         initialize weights of neural network
        #         trackname = np.random.choice(self.tracklist)
        #         self.random_playlist(self.track_num)
        self.initial_audios()
        # self.generate_original_overlays(saveto=True)
        # first_track = self.queue.pop(0)
        # self.current_track = Audio(first_track)
        # self.current_track.construct_frames()
        # self.current_frames = self.current_track.audio_frames[:FRAMES_PER_STEP] # current state
        # self.next_frames = self.current_track.audio_frames[FRAMES_PER_STEP:FRAMES_PER_STEP+10] # partial observation
        # self.current_t1framesID = 0

    def set_rl_agent(self, rl_agent):
        self.rl_agent = rl_agent

    def generate_original_overlays(self, wavname="original", saveto=False):
        nums = len(self.playlist)
        position = 10
        channels = np.zeros(global_sr * (position * nums + 5), dtype="float32")
        self.frame_matrix = []
        for i, track in enumerate(self.playlist):
            tmp_aud = Audio(track)
            tmp_aud.construct_frames()
            ones = np.ones(tmp_aud.frames_num)
            tmp_arr = np.array([])
            for frames in tmp_aud.audio_frames:
                tmp_arr = np.append(tmp_arr, frames)
            padding = (
                int(global_sr * position * i, ),
                int(global_sr * position * (nums - i - 1))
            )
            mpadding = (
                int(10 * position * i, ),
                int(10 * position * (nums - i - 1))
            )
            tmp_arr = np.pad(tmp_arr, padding, "constant")

            ones = np.pad(ones, mpadding, "constant")
            self.frame_matrix.append(ones)
            channels += tmp_arr

        self.frame_matrix = np.array(self.frame_matrix)
        channels.astype("float32")
        self.original_copy = channels
        self.original = channels
        # ipd.Audio(channels, rate=global_sr)
        if saveto:
            write(f"outputs/{wavname}.wav", global_sr, channels)

    #             audio_segment.export("original.wav", format="wav")

    def generate_new_overlays(self, wavname="new", saveto=False):
        nums = len(self.playlist)
        position = 10
        channels = np.zeros(global_sr * (position * nums + 5), dtype="float32")
        # self.frame_matrix = []
        print(len(self.audios))
        for i, track in enumerate(self.audios):
            # ones = np.ones(track.frames_num)
            tmp_arr = np.array([])
            for frames in track.audio_frames:
                tmp_arr = np.append(tmp_arr, frames)
            padding = (
                int(global_sr * position * i, ),
                int(global_sr * position * (nums - i - 1))
            )
            # print(f"{i} - {padding}")

            tmp_arr = np.pad(tmp_arr, padding, "constant")
            # print(tmp_arr.shape)
            # print(channels.shape)
            channels += tmp_arr
        if saveto:
            filename = f"outputs/{wavname}.wav"
            write(filename, global_sr, channels)
            print(f"{filename} SAVED")

    def waveplot_original_overlays(self, total=False):
        nums = len(self.playlist)
        position = 10
        channels = []
        all_channels = np.zeros(global_sr * (position * nums + 5), dtype="float32")

        for i, track in enumerate(self.playlist):
            tmp_aud = Audio(track)
            tmp_aud.construct_frames()
            tmp_arr = np.array([])
            for frames in tmp_aud.audio_frames:
                tmp_arr = np.append(tmp_arr, frames)
            padding = (
                int(global_sr * position * i, ),
                int(global_sr * position * (nums - i - 1))
            )
            tmp_arr = np.pad(tmp_arr, padding, "constant")
            #             tmp_arr = librosa.amplitude_to_db(tmp_arr)
            channels.append(tmp_arr)
            all_channels += tmp_arr

        plt.figure(figsize=(15, 5), dpi=120)
        colors = ["c", "m", "y", "b", "g", "r"]
        for c in channels[::1]:
            kwargs = {"color": colors.pop(0)}
            #             plt.plot(c, **kwargs)
            librosa.display.waveplot(c, global_sr, alpha=0.75, **kwargs)
        if total:
            kwargs = {"color": "black"}
            librosa.display.waveplot(all_channels, global_sr, alpha=0.25, **kwargs)

    def waveplot_compare(self, total=False):
        nums = len(self.playlist)
        position = 10
        channels, channels1 = [], []
        all_channels = np.zeros(global_sr * (position * nums + 5), dtype="float32")
        all_channels1 = np.zeros(global_sr * (position * nums + 5), dtype="float32")

        for i, track in enumerate(self.playlist):
            tmp_aud = Audio(track)
            tmp_aud.construct_frames()
            tmp_arr = np.array([])
            for frames in tmp_aud.audio_frames:
                tmp_arr = np.append(tmp_arr, frames)
            padding = (
                int(global_sr * position * i, ),
                int(global_sr * position * (nums - i - 1))
            )
            tmp_arr = np.pad(tmp_arr, padding, "constant")
            #             tmp_arr = librosa.amplitude_to_db(tmp_arr)
            channels.append(tmp_arr)
            all_channels += tmp_arr

        for i, track in enumerate(self.audios):
            tmp_arr = np.array([])
            for frames in tmp_aud.audio_frames:
                tmp_arr = np.append(tmp_arr, frames)
            padding = (
                int(global_sr * position * i, ),
                int(global_sr * position * (nums - i - 1))
            )
            tmp_arr = np.pad(tmp_arr, padding, "constant")
            #             tmp_arr = librosa.amplitude_to_db(tmp_arr)
            channels1.append(tmp_arr)
            all_channels1 += tmp_arr

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=120)

        colors = ["c", "m", "y", "b", "g", "r"]
        for i in range(5):
            kwargs = {"color": colors.pop(0)}
            #             plt.plot(c, **kwargs)
            ax1.set_ylim(-1, 1)
            ax1.plot(channels[i], **kwargs)
            #             librosa.display.waveplot(channels[i], global_sr, alpha=0.75, **kwargs)
            ax2.set_ylim(-1, 1)
            ax2.plot(channels1[i], **kwargs)

        #             librosa.display.waveplot(channels1[i], global_sr, alpha=0.75, **kwargs)

        # plt.show()
        # plt.savefig(f"{graph_folder}a.png")
        print("Figure shown")
        if total:
            kwargs = {"color": "black"}
            librosa.display.waveplot(all_channels, global_sr, alpha=0.25, **kwargs)

    def update_active_tracks(self):
        current = self.C
        column = self.frame_matrix[:, current]
        playing_index = np.where(column == 1)[0]
        self.active_tracks = playing_index
        # print(f"current c: {current}, \nActive track: {self.active_tracks}")
        return 1

    def update_C(self):
        self.C += 1
        return 1

    def latest_audio_frame(self):
        #         self.current_track.frames[:]
        pass

    def get_state(self):
        self.update_active_tracks()
        active_point = self.C
        active_tracks = self.active_tracks

        a, *b = active_tracks
        ib, fb = None, None
        fa = active_point - (a * 100)
        if len(b) > 0:
            ib = b[0]
            fb = active_point - (ib * 100)
            self.both_playing = True
        if len(b) == 0:
            self.both_playing = False
        self.mloc = [a, ib, fa, fb]

        self.current_track = self.audios[a]
        self.current_frame = self.audios[a].audio_frames[fa]
        if ib is not None:
            self.current_trackb = self.audios[ib]
            self.current_frameb = self.audios[ib].audio_frames[fb]
        else:
            self.current_trackb = self.audios[a]
            if fa + 1 < 149:
                self.current_frameb = self.audios[a].audio_frames[fa + 1]
            else:
                self.current_frameb = self.audios[a].audio_frames[fa]

        return [self.current_frame, self.current_frameb]

    def update_state(self, verbose=False):
        """
        Outputs are locations(index) of current audio frames
        """
        self.update_active_tracks()
        active_point = self.C
        active_tracks = self.active_tracks

        a, *b = active_tracks
        ib, fb = None, None
        fa = active_point - (a * 100)
        # (f"B{b}")
        if len(b) > 0:
            ib = b[0]
            fb = active_point - (ib * 100)
            self.both_playing = True
            # print(f"SBP:{self.both_playing}")
        if len(b) == 0:
            self.both_playing = False
        self.mloc = [a, ib, fa, fb]

        if verbose:
            print(f"""
            current track: {a}
            current frame: {fa}
            current trackB: {ib}
            current frameB: {fb}
            """)
        self.current_track = self.audios[a]
        self.current_frame = self.audios[a].audio_frames[fa]
        if ib is not None:
            self.current_trackb = self.audios[ib]
            self.current_frameb = self.audios[ib].audio_frames[fb]
        else:
            self.current_trackb = self.audios[a]
            if fa + 1 < 149:
                self.current_frameb = self.audios[a].audio_frames[fa + 1]
            else:
                self.current_frameb = self.audios[a].audio_frames[fa]

        return [
            self.current_track,
            self.current_frame,
            self.current_trackb,
            self.current_frameb,
        ]

    def update_observation(self, new_state):
        self.curr_observation = [None, None]
        a, ib, fa, fb = self.mloc
        if self.both_playing:
            # self.curr_observation[0] = self.audios[a].audio_frames[fa - 2 : fa + 3]
            self.curr_observation[0] = new_state[0]
            #             self.curr_pitches[0] = estimate_pitch(self.audios[a].audio_frames[fa], sr=global_sr)
            #             self.curr_pitches[1] = estimate_pitch(self.audios[ib].audio_frames[fb], sr=global_sr)
            #             self.curr_amp1 = self.audios[a].audio_frames[fa]
            #             self.curr_amp1 = self.audios[a].audio_frames[fa]
            if ib:
                # self.curr_observation[1] = self.audios[ib].audio_frames[fb]
                self.curr_observation[1] = new_state[1]
            #                 self.curr_amp2 = self.audios[ib].audio_frames[fb]
            else:
                self.curr_observation[1] = self.curr_observation[0]
        else:
            self.curr_observation = [-1, -1]
        return

    def update_before_after_observation(self, number=3):
        self.curr_observation = [None, None]
        a, ib, fa, fb = self.mloc
        surrounding_frames = number
        if self.both_playing:
            frames = None
            tmp_arr = self.audios[a].audio_frames
            if fa == 0:
                frames = tmp_arr[:surrounding_frames]
            else:
                frames = np.roll(tmp_arr, -(fa - 1), axis=0)[:surrounding_frames]
            self.curr_observation[0] = join_frames(frames)
            if ib:
                framesb = None
                tmp_arr = self.audios[ib].audio_frames
                if ib == 0:
                    framesb = tmp_arr[:surrounding_frames]
                else:
                    framesb = np.roll(tmp_arr, -(fb - 1), axis=0)[:surrounding_frames]
                self.curr_observation[1] = join_frames(framesb)
            else:
                self.curr_observation[1] = self.current_observation[0].copy()
        else:
            self.curr_observation = [-1, -1]
        return

    def set_action(self, action_id):
        if self.both_playing:
            if action_id < 0:
                print(f"action id: {action_id} unknown")
                return self.actions.no_action
            next_action = self.actions.action_dict[str(action_id)]
            self.action_to_exec = next_action
            return next_action
        else:
            self.action_to_exec = self.actions.no_action
            return self.actions.no_action

    def set_random_action(self):
        if self.both_playing:
            ate = np.random.choice([
                self.actions.increase_pitch,
                self.actions.decrease_pitch,
                #                 self.actions.increase_frame_volume,
                #                 self.actions.decrease_frame_volume,
            ])
            self.action_to_exec = ate

            return ate
        else:
            self.action_to_exec = self.actions.no_action
            return self.actions.no_action

    def get_action_id(self):
        aid = self.actions.obtain_action_id(self.action_to_exec)
        return aid

    def update_action_history(self, track_index, frame_index, action_val):
        self.audios[track_index].action_history[frame_index] = action_val

    def update_one_step(self):
        state_ab = [state_a, state_b] = self.get_state()
        state_a_next = None
        # rp, rv, rp2, rv2 = self.update_state()
        s1, f1, s2, f2 = self.update_state()
        action_val = 0
        mdval = 0
        # print(f"action val: {action_val} / {mdval}")
        if self.both_playing:
            actionID = 5
            action_val = self.rl_agent.explore_action_val(state=state_ab)[0]
            mdval = np.floor(16 * action_val) - 8
            param = {"pitch_change": mdval}
            print(f"actionval: {action_val} \ p: {param}")
            self.set_action(actionID)
            state_a_next = self.exec_action(self.action_to_exec, **param)
        else:
            actionID = 0
            self.set_action(actionID)
            self.exec_action(self.action_to_exec)
            state_a_next = state_ab
        self.update_C()
        self.update_action_history(self.mloc[0], self.mloc[2], action_val)

        s1x, f1x, s2x, f2x = self.update_state()

        self.rewarder.set_observation([-1, -1])
        self.update_observation(new_state=state_a_next)

        if not self.both_playing:
            return 0
        self.rewarder.set_observation(state_a_next)
        step_reward = self.rewarder()
        print(f"step reward: {step_reward}")
        if self.both_playing:
            self.rl_agent.memory.push(state_ab, action_val, state_a_next, step_reward)

        return step_reward

    def observe(self):
        """
        Observe at current state and push to replay memory
        """
        if not self.both_playing:
            return 0
        if self.both_playing:
            step_reward = self.reward()
            self.rl_agent.memory.push(state_ab, action_val, state_a_next, step_reward)
        return 1

    def exec_action(self, func, **param):
        new_frame = func(self.current_frame, **param)
        a, ib, fa, fb = self.mloc
        self.audios[a].alter_frame(new_frame, index=fa)
        log = (str(fa), str(func))
        self.audios[a].frame_log.append(log)
        if ib is None or fb is None:
            return new_frame
        new_frames = [new_frame, self.audios[ib].audio_frames[fb]]
        # return new_frame
        return new_frames


    def cross_fade_reward(self):
        if not self.both_playing:
            return 0
        a, ib, fa, fb = self.mloc
        cf_reward = None

        return cf_reward



    def sample_action(self):
        new_action = self.actions.sample_action()
        pass

    def load_playlist(self):
        self.playlist = self.tracklist
        self.queue = self.playlist.copy()

    def random_playlist(self, n=5):
        self.playlist = random.sample(self.tracklist, n)
        self.queue = self.playlist.copy()

    def initial_audios(self):
        self.audios = []
        for track in self.playlist:
            tmp_aud = Audio(track)
            tmp_aud.construct_frames()
            tmp_aud.estimate_pitch_for_every_frame()
            self.audios.append(tmp_aud)
        return 1

    def recover_audios(self):
        for audio in self.audios:
            audio.newest_version = audio.audio_frames.copy()
            audio.construct_frames()
        self.both_playing = False
        return 1

    def save_audios_history(self, epoch):
        for audio in self.audios:
            audio.save_current(epoch)
        return 1

    def save_amp_history(self, epoch):
        audio_avgs = []
        for audio in self.audios:
            amp_avg = []
            for f in audio.newest_version:
                amp_avg.append(f.mean())
            audio_avgs.append(amp_avg)
        np.save(f"amp/amp_epoch{epoch}", audio_avgs)

