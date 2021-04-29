import numpy as np
import librosa
from utilz import *



class Actions:
    def __init__(self):
        self.action_space = []

    #         self.load_action()
    #         self.n = len(action_space)

    def get_action_space(self):
        return self.action_space

    def increase_frame_volume(self, input_frame, level=1.1):
        input_frame = input_frame * level
        return input_frame

    def decrease_frame_volume(self, input_frame, level=0.9):
        input_frame = input_frame * level
        return input_frame

    def adjust_pitch(self, input_frame, pitch_change=0):
        bins_per_octave = 12
        new_audio = librosa.effects.pitch_shift(input_frame.astype("float64"), sr=global_sr,
                                                n_steps=pitch_change,
                                                bins_per_octave=bins_per_octave)
        return new_audio


    def exploration_action(self, eps):
        action_val = 0
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        # add noise
        noise = np.random.randn(self.action_dim) * epsilon
        action += noise
        return action


    def increase_pitch(self, input_frame, pitch_change=4):
        if pitch_change < 1:
            pitch_change = 1
        bins_per_octave = 12
        new_audio = librosa.effects.pitch_shift(input_frame.astype("float64"), sr=global_sr,
                                                n_steps=pitch_change,
                                                bins_per_octave=bins_per_octave)
        #         duration = librosa.get_duration(y=input_frame, sr=global_sr)
        #         new_duration = librosa.get_duration(y=new_audio, sr=global_sr)
        #         old_len = len(input_frame)
        #         new_len = len(new_audio)
        #         print(f"input audio duration: {old_len}\noutput audio duration: {new_len}")
        return new_audio

    def decrease_pitch(self, input_frame, pitch_change=.6):
        if pitch_change > 1:
            pitch_change = 1
        bins_per_octave = 12
        new_audio = librosa.effects.pitch_shift(input_frame.astype("float64"), sr=global_sr,
                                                n_steps=pitch_change,
                                                bins_per_octave=bins_per_octave)
        return new_audio

    def time_stretch(self, input_frame, rate=1):
        new_audio = librosa.effects.time_stretch(input_frame.astype("float64"), rate=rate)
        return new_audio

    def no_action(self, input_frame):
        return input_frame

    def load_action(self):
        self.action_space = [
            self.decrease_pitch, self.increase_pitch,
            self.decrease_frame_volume, self.increase_frame_volume,
            self.time_stretch, self.no_action,
        ]
        self.n = len(self.action_space)


    def load_action_dict(self):
        self.action_dict = {
            "0": self.no_action,
            "1": self.increase_frame_volume,
            "2": self.decrease_frame_volume,
            "3": self.increase_pitch,
            "4": self.decrease_pitch,
            "5": self.adjust_pitch,
        }
        return self.action_dict


    def obtain_action_id(self, act):
        if act == self.no_action:
            return 0
        for k in list(self.action_dict.keys()):
            if self.action_dict[k] == act:
                return int(k)
        return -1

    def sample_action(self):
        if self.action_box is None:
            self.load_action()
        sample_act = np.random.choice(self.action_box)
        return sample_act

    def __len__(self):
        return len(self.action_space)
