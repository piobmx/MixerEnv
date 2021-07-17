import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from utilz import *

class Audio:
    def __init__(self, audiofilepath):
        self.y, self.sr = librosa.load(audiofilepath, sr=global_sr)
        self.yshape = self.y.shape

        self.original = self.y.copy()
        self.newest_y = self.y.copy()
        self.duration = librosa.get_duration(y=self.y, sr=global_sr)
        self.timestamps = np.linspace(0, self.duration, int(global_sr * self.duration))
        self.frames_index = librosa.time_to_frames(self.timestamps, sr=global_sr,
                                                   hop_length=global_hop_len)
        self.frames_num = max(self.frames_index)
        self.action_box = None
        self.audio_frames = []
        self.mels = []
        self.best_reward = -1
        # self.step = 0
        self.mark = 1
        self.locked = []
        self.best_mismatches = None
        self.shft, self.stch = 0, 1
        self.output_list = []
        self.log_for_sox =[]
        self.frame_log = []
        self.action_history = [-1 for _ in range(self.frames_num)]
        self.epoch_history = {}
        self.shifted = 0
        self.stretched = 1
        self.pitched = 0
        self.vlines_widths = 0.75
        self.vlines_colors = "Blue"
        self.backward_limit = self.duration * 0.1
        self.forward_limit = self.duration * 0.1

    def __getitem__(self, item):
        return self.audio_frames[item]

    def get_original_audio(self):
        return self.original

    def construct_frames(self, construct_mels=False):
        """
        Splitting audio into frames of 4410 samples.
        Not constructing mel-spectrogram anymore
        """
        self.audio_frames = []
        self.mels = []
        for i in range(self.frames_num - 1):
            next_frame = self.y[i * frame_len: (i + 1) * (frame_len)]
            self.audio_frames.append(next_frame)
            if construct_mels:
                mel = audio_to_spec(next_frame)
                self.mels.append(mel)
        last_frame = self.y[(i + 1) * (frame_len): (i + 1) * (frame_len) + frame_len]
        self.audio_frames.append(last_frame)
        if construct_mels: self.mels.append(audio_to_spec(last_frame))
        return None

    def update_shift(self, level, backward_limit=5, forward_limit=5):
        """
        Update the time-shift factor, and set a limit to how far the audio clip can be shifted
        """
        backward_limit = self.backward_limit
        forward_limit = self.forward_limit
        shifted = self.shifted + level
        # print("selfshifted: ", self.shifted, ".", level)
        if shifted > forward_limit:
            shifted = 0.0
        elif shifted < -backward_limit:
            shifted = 0.0
        self.shifted = shifted
        return self.shifted

    def update_stretch(self, level, down_limit=2/3, up_limit=3/2):
        """
        Update the time-stretch factor, and limit how much the audio clip can be stretched.
        The default parameters mean that the at any point of the training the ratio of the duration of the new audio
        should be with 3/2 and 2/3 to the duration of the un-stretched audio.
        """
        stretch_level = 1 + level
        stretched = self.stretched * stretch_level
        if stretched > up_limit:
            stretched = 1.0
        elif stretched < down_limit:
            stretched = 1.0
        self.stretched = stretched
        return stretched

    def set_vlines_widths(self, end, step, start=0, thin=0.5, thick=1):
        """
        Data for plt plotting
        """
        ref = np.ones(self.beats.shape) * thin
        ref[start:end:step] = thick
        self.vlines_widths = ref
        return ref

    def set_vlines_colors(self, end, step, default_c="m", highlight_c="black", start=0):
        """
        Data for plotting
        """
        ref = np.ones(self.beats.shape)
        ref[start:end:step] = 2
        clrs = [highlight_c if _ == 2 else default_c for _ in ref]
        self.vlines_colors = clrs
        return clrs

    def generate_waveform_from_sox_log(self):
        output_arr = None
        print(self.log_for_sox)
        return output_arr

    def save_current(self, key):
        self.epoch_history[key] = (key, self.audio_frames)

    def alter_frame(self, new_frame, index):
        self.audio_frames[index] = new_frame
        return 1

    def estimate_pitch_for_every_frame(self):
        self.pitches = []
        for frame in self.audio_frames:
            ep = estimate_pitch(frame, sr=global_sr)
            self.pitches.append(ep)
        return 1

    def merge_all_frames(self):
        mixed = np.array([])
        for i, f in enumerate(self.audio_frames):
            mixed = np.append(mixed, f)
        return mixed

    def average_frame_volume(self, original=True):
        if original:
            for f in self.audio_frames:
                pass