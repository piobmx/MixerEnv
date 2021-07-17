from utilz import *
from Audio import Audio
from Rewarder import Rewarder
# from DQN import DQN, ReplayMemory
import datetime, os, sys, random, time

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import sox
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler

print(sys.version)
print(sys.executable)

FRAMES_PER_STEP = 10


class Mixer_agent():
    """
    Main mixer environment.
    """
    def __init__(self, tracklist, queue=[]):
        self.tracklist = tracklist
        self.track_num = 5
        self.queue = queue
        self.audios = []
        self.rl_agent = None

        self.original = None

        self.step = 0
        self.episode = 0
        # self.actions = actions
        # self.actions.load_action_dict()
        self.action_history = []
        self.best_num_in_sync = 0
        self.active_tracks = None
        self.current_track = None
        self.next_track = None
        self.observation = None
        self.current_obs = []
        self.pitch_matrix = []
        self.pitch_obs = []
        self.next_pitch_obs = []
        self.rewarder = Rewarder()
        self.both_playing = False
        # self.action_to_exec = self.actions.no_action
        self.C = 0
        self.nn_size = None
        self.scaler = StandardScaler()

    def reset(self, n=None):
        self.initial_audios(n=n)

    def reset_onsets(self):
        """
        Detect beats locations and reset
        """
        self.start_time = [0, 5, 20, 30, 40]
        for i, aud in enumerate(self.audios):
            joined = join_frames(aud.audio_frames)
            oe, beats, times = detect_onsets(joined)
            aud.Y = aud.y.copy()
            beats = np.insert(beats, 0, 0)
            self.nn_size = len(beats)
            aud.beats = beats
            aud.onsets = oe
            aud.beat_times = times
            bis = aud.beat_times[aud.beats]
            aud.shifted = 0
            aud.stretched = 1
            aud.mark = 1
            aud.locked = []
            aud.output_list = []
            aud.log_for_sox = []
            aud.best_reward = 0
            # bis = np.insert(bis, 0, 0)
            # bis = np.append(bis, 15.0)
            aud.beats_in_seconds = bis.copy()
            aud.BEATS = bis.copy()
            aud.newest_bts = bis.copy()
            aud.backup_bts = bis.copy()
            aud.backup_y = aud.y.copy()
            aud.newest_y = aud.y.copy()
            aud.abs_beats_in_seconds = aud.beats_in_seconds + self.start_time[i]
            aud.B = aud.abs_beats_in_seconds.copy()
            aud.vlines_widths = np.ones(beats.shape)
            aud.best_beat_match = np.ones(aud.B.shape) * 1000
            aud.best_result = aud.abs_beats_in_seconds.copy()
            aud.best_mismatches_sum = 1000
        abis = self.audios[0].abs_beats_in_seconds

        for i, aud in enumerate(self.audios[1:]):
            s=sox.Transformer(); s.pad(start_duration=5)
            aud.newest_y = soxx(s, aud.newest_y)
            next_abis = aud.abs_beats_in_seconds
            dis_arr = find_nearest(abis, next_abis)
            aud.dis_arr = dis_arr
            abis = next_abis

    def episodic_onset_reset(self):
        """
        Reset onset data episodically during the training
        """
        self.audios[1].best_reward = 0
        self.audios[1].best_mismatches = []
        self.audios[1].BEATS = self.audios[1].beats_in_seconds.copy()
        self.audios[1].Y = self.audios[1].y.copy()
        return

    def reset_pitch_by_frame(self):
        self.C = 0
        maf = self.audios[0].audio_frames.copy()
        mbf = self.audios[1].audio_frames.copy()
        self.MAX_C = len(mbf)
        self.distance = np.zeros(self.MAX_C)
        self.chroma_a = []
        self.chroma_b = []
        self.CHROMA_A = []
        self.CHROMA_B = []
        self.best_modulated = maf.copy()
        self.y_frames = maf.copy()
        self.distance = []
        for i, frame in enumerate(maf):
            chroma_a = librosa.feature.chroma_stft(frame, sr=global_sr, hop_length=4410, center=False)
            chroma_b = librosa.feature.chroma_stft(mbf[i], sr=global_sr, hop_length=4410, center=False)
            # max_chroma_a = np.argmax(chroma_a, axis=0)
            # max_chroma_b = np.argmax(chroma_b, axis=0)
            dis = np.linalg.norm(chroma_a - chroma_b)
            self.distance.append(dis)
            self.CHROMA_A.append(chroma_a)
            self.CHROMA_B.append(chroma_b)
            self.chroma_a.append(chroma_a.flatten())
            self.chroma_b.append(chroma_b.flatten())
            # dis = np.linalg.norm(chroma_a - chroma_b)
            # self.distance[i] = dis
        self.CHROMA_A_PLOT = np.concatenate(self.CHROMA_A, axis=1)
        self.CHROMA_B_PLOT = np.concatenate(self.CHROMA_B, axis=1)
        self.chroma = self.chroma_a.copy()
        self.input_dim = 12
        self.distance_b = self.distance.copy()
        self.action_dim = 1
        self.frame_pitch_value = np.zeros(self.MAX_C).astype(np.float64)
        self.epoch_action_log = []
        self.best_dist_log = np.zeros(self.MAX_C) - 100
        self.best_action_log = np.ones(self.MAX_C) * 100
        self.best_log = self.chroma_a.copy()
        self.best_pitch_score = -1
        self.default_state = self.chroma_a[0] - self.chroma_b[0]
        return self.input_dim, self.action_dim

    def episodic_reset_pitch_by_frame(self):
        """
        Reset pitch and chroma information episodically during the training
        """
        self.C = 0
        self.distance = self.distance_b
        self.chroma = self.chroma_a.copy()
        self.frame_pitch_value = np.zeros(self.MAX_C).astype(np.float64)
        self.epoch_action_log = []
        self.best_dist_log = np.zeros(self.MAX_C) - 100
        self.best_action_log = np.ones(self.MAX_C) * 100
        self.best_log = self.chroma_a.copy()
        self.best_pitch_score = -1
        self.best_modulated = self.y_frames.copy()
        default_state = self.chroma_a[0] - self.chroma_b[0]

    def get_next_pre_state(self):
        if self.C > len(self.chroma) - 1:
            return None
        old_state = self.chroma[self.C]
        ref_state = self.chroma_b[self.C]
        return old_state - ref_state

    def set_rl_agent(self, rl_agent):
        self.rl_agent = rl_agent

    def create_frame_matrix(self):
        nums = len(self.playlist)
        position = 10
        channels = np.zeros(global_sr * (position * nums + 5), dtype="float32")
        self.frame_matrix = []
        for i, track in enumerate(self.playlist):
            tmp_aud = Audio(track)
            tmp_aud.construct_frames()
            ones = np.ones(tmp_aud.frames_num)
            tmp_arr = np.array([])
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

    def update_C(self):
        self.C += 1
        return 1

    def sample_beats_action_space(self, lims, action_num=1):
        return np.random.uniform(lims[0], lims[1], size=(1, action_num))

    def plot_original_beats(self, eval_dir, ax=None):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=True)
        custom_xlim = (-1, 1 + self.audios[1].abs_beats_in_seconds.max())
        custom_ylim = (-1.1, 1.1)
        trackA, trackB = self.audios[0].y, self.audios[1].y
        B0, B1 = self.audios[0].abs_beats_in_seconds, self.audios[1].abs_beats_in_seconds

        l0, l1 = len(B0), len(B1)
        ymin = -self.audios[1].onsets[self.audios[1].beats]
        colors0 = ["green" if _ % 2 == 1 else "red" for _ in range(l0)]
        colors1 = ["green" if _ % 2 == 1 else "red" for _ in range(l1)]

        librosa.display.waveplot(trackA, sr=global_sr, alpha=.75, ax=ax[0])
        ax[0].vlines(B0, ymin=-self.audios[0].onsets[self.audios[0].beats]/20,
                     ymax=self.audios[0].onsets[self.audios[0].beats]/20,
                     alpha=0.85, color=colors0, linestyle=':', label='Beats',
                     linewidth=3, )

        padsox = sox.Transformer();
        padsox.pad(start_duration=self.start_time[1])
        trackB = padsox.build_array(input_array=trackB, sample_rate_in=global_sr)
        librosa.display.waveplot(trackB, sr=global_sr, alpha=.75, ax=ax[1])
        ax[1].vlines(B1, ymin=-self.audios[1].onsets[self.audios[1].beats]/20,
                     ymax=self.audios[1].onsets[self.audios[1].beats]/20,
                     alpha=0.85, color=colors1, linestyle=':', label='Beats',
                     linewidth=3, )

        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        plt.savefig(eval_dir + f"original.png")
        plt.close()
        return

    def mixing_tracks(self, T1, T2, dir, fadeout1):
        fad1 = sox.Transformer()
        fad1.fade(fade_out_len=fadeout1, fade_shape='l')
        fad2 = sox.Transformer()
        fad2.fade(fade_in_len=fadeout1, fade_out_len=1.5, fade_shape="l")
        path1, path2 = "testsox/tmp1.wav", "testsox/tmp2.wav"
        pathx = dir
        T1 = fad1.build(input_array=T1.copy(), sample_rate_in=global_sr,
                        output_filepath=path1)
        T2 = fad1.build(input_array=T2.copy(), sample_rate_in=global_sr,
                        output_filepath=path2)

        cbn = sox.combine.Combiner()
        cbn.build(
            input_filepath_list=[path1, path2],
            output_filepath=pathx,
            combine_type="mix"
        )
        return

    def plot_pitches(self, eval_dir, step, to_plot=None):
        fig, ax = plt.subplots(4, 1, figsize=(12, 8), dpi=240)
        # ch = [x.reshape(12, 1) for x in self.chroma]
        if to_plot is not None:
            ch = to_plot
        else:
            ch = [x.reshape(12, 1) for x in self.best_log]
        new_chroma = np.concatenate(ch, axis=1)
        librosa.display.specshow(self.CHROMA_A_PLOT, y_axis="chroma", x_axis="time",
                                 cmap='coolwarm', ax=ax[1])
        librosa.display.specshow(self.CHROMA_B_PLOT, y_axis="chroma", x_axis="time",
                                 cmap='coolwarm', ax=ax[0])
        librosa.display.specshow(new_chroma, y_axis="chroma", x_axis="time", cmap='coolwarm', ax=ax[2])
        ax[-1].hist(self.epoch_action_log, density=True, facecolor='g', alpha=0.75)
        plt.savefig(eval_dir + f"{step}.png")
        plt.close()
        return

    def plot_newest_beats(self, eval_dir, step):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=True)
        custom_xlim = (-1, 3 + self.audios[1].abs_beats_in_seconds.max())
        custom_ylim = (-1.1, 1.1)
        trackA, trackB = self.audios[0].y, self.audios[1].Y
        B0, B1 = self.audios[0].beats_in_seconds, self.audios[1].BEATS

        l0, l1 = len(B0), len(B1)
        ymin = -self.audios[1].onsets[self.audios[1].beats]
        colors0 = self.audios[0].vlines_colors
        colors1 = self.audios[1].vlines_colors
        if np.random.random() < 0.01:
            write(eval_dir+f"{step}A.wav", 44100, trackA)
            write(eval_dir+f"{step}B.wav", 44100, trackB)
            np.save(eval_dir+f"{step}B0", B0, )
            np.save(eval_dir+f"{step}B1", B1, )
            fd1 = B1[0]
            self.mixing_tracks(trackA, trackB, dir=eval_dir+f"{step}.wav", fadeout1=fd1)

        librosa.display.waveplot(trackA, sr=global_sr, alpha=.45, ax=ax[0])
        ax[0].vlines(B0, ymin=-self.audios[0].onsets[self.audios[0].beats]/20,
                     ymax=self.audios[0].onsets[self.audios[0].beats]/20,
                     alpha=1, color=colors0, linestyle=':', label='Beats',
                     linewidth=1, )

        padsox = sox.Transformer()
        # padsox.pad(start_duration=(self.audios[1].start_duration))
        trackB = padsox.build_array(input_array=trackB, sample_rate_in=global_sr)
        librosa.display.waveplot(trackB, sr=global_sr, alpha=.45, ax=ax[1])
        yminx = -self.audios[1].onsets[self.audios[1].beats]/20
        # ymaxx = self.audios[1].onsets[self.audios[1].beats]/20
        ax[1].vlines(B1, ymin=yminx, ymax=-yminx,
                     alpha=1, color=colors1, linestyle=':', label='Beats',
                     linewidth=self.audios[1].vlines_widths)

        mismatch = self.audios[1].best_mismatches
        for i, m in enumerate(mismatch):
            # if i % 2 == 0: c="red"
            # else: c = "green"
            c = "green"
            if colors1[i] == "black":
                c = "red"
            if np.abs(m) < self.beat_thr:
                c = "blue"
            ax[1].axvspan(xmin=B1[i]-m, xmax=B1[i]+m, ymin=yminx[i], ymax=1.8, alpha=0.25,
                          color=c, zorder=0, clip_on=False)
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        plt.savefig(eval_dir + f"{step}.png")
        plt.close()
        return

    def live_pitch_match(self, action_value, update_live=True, use_sox=True):
        new_state, env_reward, done = [], 0, False
        ref_state = self.chroma_b[self.C]
        old_state = self.chroma[self.C]
        pitch_step = action_value[0]

        # if pitch_step > 1:
        #     pitch_step = 1
        # if pitch_step < -1:
        #     pitch_step = -1
        self.epoch_action_log.append(pitch_step)
        if use_sox:
            pitchsox = sox.Transformer();
            norm_level = 6 * pitch_step
            pitchsox.pitch(norm_level, quick=True)
            old_frame = self.audios[0].audio_frames[self.C]
            modulated = soxx(pitchsox, old_frame)
            new_state = librosa.feature.chroma_stft(modulated,
                                                    sr=global_sr,
                                                    hop_length=4410,
                                                    center=False)
            self.best_modulated[self.C] = modulated
            new_state = new_state.flatten()
        else:
            norm_level = np.int_(6 * pitch_step)
            new_state = np.roll(old_state, norm_level)
        # print(new_state)
        # print(ref_state)
        max_index_1 = np.argmax(new_state)
        max_index_2 = np.argmax(ref_state)
        # if np.argmax(ref_state) == np.argmax(new_state):
        distance = np.linalg.norm(new_state - ref_state)
        if max_index_2 == max_index_1 or np.abs(max_index_2 - max_index_1) == 5 \
                or np.abs(max_index_2 - max_index_1) == 7:
            env_reward = 1
        else:
            env_reward = -distance
            env_reward = np.clip(env_reward, -1, 1)

        if env_reward > self.best_dist_log[self.C]:
            self.best_dist_log[self.C] = env_reward
            self.best_log[self.C] = new_state
            self.best_action_log[self.C] = norm_level

        # print(env_reward)
        if update_live: self.chroma[self.C] = new_state
        next_state = new_state - ref_state
        return next_state, env_reward, done, distance

    def plot_newest_pitches(self, eval_dir, step):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=240)
        ax[0].plot(np.arange(self.chroma_a.size), self.chroma_a_, linewidth=.3, color="red")
        ax[0].plot(np.arange(self.best_pitch.size), self.best_pitch, linewidth=1, color="black")
        ax[0].plot(np.arange(self.chroma_b.size), self.chroma_b, linewidth=.3, color="blue")
        diff = np.abs(self.chroma_b - self.best_pitch)
        diff[diff==7] = 100
        diff[diff==5] = 100
        diff[diff==0] = 100
        diff[diff!=100] = 0
        diff[diff==100] = 1
        ax[0].plot(np.arange(self.chroma_b.size), diff, color="green", linewidth=6)
        axions = np.array(self.epoch_action_log)
        a1 = axions[:, 1]
        a0 = axions[:, 0]
        n, bins, patches = ax[1].hist(a1, 50, density=True, facecolor='g', alpha=0.75)
        n, bins, patches = ax[2].hist(a0, 50, density=True, facecolor='g', alpha=0.75)

        plt.savefig(eval_dir + f"{step}.png")
        plt.close()
        return 0

    def plot_explored_action_space(self, eval_dir, step):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=240)
        axions = np.array(self.epoch_action_log)
        a1 = axions[:, 1]
        a0 = axions[:, 0]
        n, bins, patches = ax[0].hist(a1, 50, density=True, facecolor='g', alpha=0.75)
        n, bins, patches = ax[1].hist(a0, 50, density=True, facecolor='g', alpha=0.75)
        plt.savefig(eval_dir + f"{step}.png")
        plt.close()
        return

    def beat_match(self, action_value, step=2):
        new_state, reward, done = [], 0, False
        best_n = np.int_(self.nn_size * (2/3))
        step = step
        static = self.audios[0].beats_in_seconds.copy()
        sdiff = static[-1] / 3
        start_time = static[-1] - sdiff
        oe, beat,  = self.audios[1].onsets, self.audios[1].beats.copy()
        beats_pos = self.audios[1].beats_in_seconds.copy()
        y0 = self.audios[1].original.copy()
        old_beats_position = beats_pos + start_time
        mismatch_arr0, _ = find_nearest(static, old_beats_position, forward=False)

        if action_value.size == 1:
            shift, stretch = action_value, None
        else:
            action_value = action_value.reshape((action_value.size, ))
            shift, stretch = action_value[0], action_value[1]

        shift = self.audios[1].update_shift(level=shift * 5)
        stretch = self.audios[1].update_stretch(level=stretch)
        # print(f"shift for {shift}, stretch for {stretch}")

        new_beats_position = (beats_pos + start_time + shift) * (stretch)
        mismatch_arr2, ind_a = find_nearest(static, new_beats_position, forward=False)
        self.audios[1].newest_bts = new_beats_position

        mismatches = mismatch_arr2[1:best_n:step]
        if self.audios[1].best_mismatches is None:
            self.audios[1].best_mismatches = mismatches
        self.beat_thr = 0.02
        self.audios[1].set_vlines_widths(thin=1, thick=3, start=1, end=best_n, step=step)
        self.audios[1].set_vlines_colors(start=1, end=best_n, step=step)
        lenB1 = len(self.audios[0].beats)
        clrs0 = ["red" for _ in range(lenB1)]
        for i, ind in enumerate(ind_a[1:best_n:step]):
            if np.abs(mismatches[i]) < self.beat_thr:
                clrs0[int(ind)] = "blue"
            else:
                clrs0[int(ind)] = "black"
        self.audios[0].vlines_colors = clrs0
        self.num_in_sync = 0
        for i, m in enumerate(mismatches):
            if np.abs(m) < self.beat_thr:
                self.num_in_sync += 1
                # reward += oe[beat[i]]
                # reward += 1
            # else:
            #     break
        if self.num_in_sync == len(mismatches):
            print("DONE")
            done = True
        reward = self.num_in_sync / len(mismatches)
        # reward = reward / 30
        if reward > self.audios[1].best_reward:
            # self.audios[1].best_mismatches_sum = mismatches_sum
            self.audios[1].best_mismatches = mismatch_arr2
            self.audios[1].best_reward = reward
            self.best_num_in_sync = self.num_in_sync
            self.audios[1].BEATS = new_beats_position
            print(f"REWARD: {reward}, stretch: {stretch}, shifh: {shift}, @ timestep: {self.episode}")
            print(f"{self.num_in_sync}, of {len(mismatches)}")
            axion = sox.Transformer()
            ax2 = sox.Transformer()
            axion.pad(start_duration=stretch*(start_time + shift))
            self.audios[1].start_location = (start_time + shift)*stretch
            if 0.9 <= stretch < 1.2:
                axion.stretch(stretch)
                ax2.stretch(stretch)
            else:
                axion.tempo(1 / stretch)
                ax2.tempo(1/ stretch)
            output = soxx(axion, y0)
            xoutput = soxx(ax2, y0)
            del axion
            self.audios[1].Y = output
            self.audios[1].Y_1 = xoutput
        else:
            reward = 0
        ########################################
        new_state = mismatch_arr2
        return new_state, reward, done

    def load_playlist(self, tracklist=None):
        if tracklist:
            self.playlist = tracklist
        else:
            self.playlist = self.tracklist
        self.queue = self.playlist.copy()

    def initial_audios(self, n=None):
        self.audios = []
        for track in self.playlist[:n]:
            tmp_aud = Audio(track)
            tmp_aud.construct_frames()
            # tmp_aud.estimate_pitch_for_every_frame()
            self.audios.append(tmp_aud)
        return 1
