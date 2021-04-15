import librosa
import numpy as np
from utilz import *


class Audio:
    def __init__(self, audiofilepath):
        self.y, self.sr = librosa.load(audiofilepath, sr=global_sr)
        self.original = self.y.copy()
        self.duration = librosa.get_duration(y=self.y, sr=global_sr)
        self.timestamps = np.linspace(0, self.duration, int(global_sr * self.duration))
        self.frames_index = librosa.time_to_frames(self.timestamps, sr=global_sr,
                                                   hop_length=global_hop_len)
        self.frames_num = max(self.frames_index)
        self.action_box = None
        self.audio_frames = []
        self.step = 0
        self.frame_log = []
        self.epoch_history = {}

    def get_original_audio(self):
        return self.original

    def construct_frames(self):
        self.audio_frames = []
        for i in range(self.frames_num - 1):
            next_frame = self.y[i * frame_len: (i + 1) * (frame_len)]
            self.audio_frames.append(next_frame)
        self.audio_frames.append(self.y[(i + 1) * (frame_len):])

        return None

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

    def random_actions(self):
        """
        Execute random action to every frame
        """
        if len(self.audio_frames) == 0:
            self.construct_frames()

        actions = Actions()
        mixed = np.array([])
        for i, f in enumerate(self.audio_frames):
            c = np.random.randint(0, 5)
            if c == 0:
                mixed = np.append(mixed, f)
            elif c == 1:
                level = np.random.randint(1, 12)
                df = actions.decrease_frame_volume(f, level=level)
                mixed = np.append(mixed, df)

            elif c == 2:
                level = np.random.randint(1, 3) * np.random.random()
                df = actions.increase_pitch(f, pitch_change=level)
                mixed = np.append(mixed, df)

            elif c == 3:
                level = np.random.random()
                df = actions.decrease_pitch(f, pitch_change=level)
                mixed = np.append(mixed, df)

            elif c == 4:
                level = np.random.randint(-12, 0)
                df = actions.decrease_frame_volume(f, level=level)
                mixed = np.append(mixed, df)

            else:
                mixed = np.append(mixed, f)

        return mixed

    def merge_all_frames(self):
        mixed = np.array([])
        for i, f in enumerate(self.audio_frames):
            mixed = np.append(mixed, f)
        return mixed


if __name__ == "__main__":
    testwav = "/Users/wxxxxxi/Projects/ReinL/test_folder/wavs/Diversion_p0.wav"
    a = Audio(testwav)
