from utilz import *
import numpy as np
import librosa

class Rewarder:
    """
    An object giving reward
    """

    def __init__(self, observation=None):
        # self.mixer = mix
        self.observation = observation
        self.pitchA = None
        self.pitchB = None

    def __str__():
        pass

    def __call__():
        return self.reward()

    def reward(self):
        preward = self.pitch_reward()
        # preward = self.amp_reward()
        return preward

    def pitch_reward(self):
        preward = 0
        # print(f"obs: {self.observation}")
        self.pitchA = estimate_pitch(self.observation[0], sr=global_sr)
        self.pitchB = estimate_pitch(self.observation[1], sr=global_sr)
        self.midiA = int(librosa.hz_to_midi(self.pitchA))
        self.midiB = int(librosa.hz_to_midi(self.pitchB))
        pdiff = np.abs(self.midiA - self.midiB)
        if pdiff % 12 == 0:
            preward = 1
        else:
            if (pdiff % 12) % 2 == 0:
                preward = 0.7
            elif (pdiff % 12) % 2 == 1:
                preward = (np.random.random() - 1) / 10
            else:
                preward = 0
        return preward

    def amp_reward(self):
        ampreward = 0
        b, p, a = self.observation[0][:4410], self.observation[0][4410:4410*2], self.observation[0][4410*2:4410*3]
        if np.average(b) > np.average(p):
            ampreward += 0.5
        else:
            ampreward -= 0.1
        if np.average(p) < np.average(a):
            ampreward += 0.5
        else:
            ampreward -= 0.1
        if np.average(b) > np.average(a):
            ampreward += 0.3
        else:
            ampreward -= 0.1
        return ampreward