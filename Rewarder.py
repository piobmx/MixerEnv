from utilz import *
import numpy as np
import librosa

class Rewarder:
    """
    An object determining the reward.
    """

    def __init__(self):
        # self.mixer = mix
        self.observation = [-1, -1]
        self.pitchA = None
        self.pitchB = None

    def __str__(self):
        pass

    def __call__(self):
        return self.reward()

    def reward(self):
        preward = self.pitch_reward()
        # preward = self.amp_reward()
        return preward

    def set_observation(self, observation):
        self.observation[0] = observation[0]
        self.observation[1] = observation[1]
        return

    def pitch_reward(self):
        """
        Receives the Agent's current observation. Analysis the pitch of both ongoing tracking.
        Reward of 1 is given if the pitches of track A and track B are 12 or 7 semitones from each other (which
        can be seen as an octave or perfect fifth so is sounds harmonious.)
        Reward of 0 is given if octave or perfect fifth doesn't happen. And in this case the Actor is supposed to learn
        to modulate the audio.
        """
        preward = 0
        # print(f"obs: {self.observation}")

        self.pitchA = estimate_pitch(self.observation[0], sr=global_sr)
        self.pitchB = estimate_pitch(self.observation[1], sr=global_sr)
        self.midiA = int(librosa.hz_to_midi(self.pitchA))
        self.midiB = int(librosa.hz_to_midi(self.pitchB))
        pdiff = np.abs(self.midiA - self.midiB)
        if pdiff % 12 == 0 or pdiff % 7 == 0:
            preward = 1
        else:
            # Other rewards are abandoned temporarily

            # if (pdiff % 12) % 2 == 0:
            #     preward = 0.7
            # elif (pdiff % 12) % 2 == 1:
            #     preward = (np.random.random() - 1) / 10
            # else:
            #     preward = 0
            preward = 0
        return preward

    def amp_reward(self):
        """
        Amplitude reward is still in progress...
        """

        ampreward = 0
        b, p, a = self.observation[0][:4410], self.observation[0][4410:4410*2], self.observation[0][4410*2:4410*3]
        if np.average(b) > np.average(p):
            ampreward += 1
        # else:
            # ampreward -= 0.1
        if np.average(p) < np.average(a):
            ampreward += 0.8
        # else:
        #     ampreward -= 0.1
        if np.average(b) > np.average(a):
            ampreward += 1
        # else:
        #     ampreward -= 0.1
        return ampreward