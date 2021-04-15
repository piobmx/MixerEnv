from mixer_agent import Mixer_agent
from utilz import *
from Actions import Actions

import numpy as np
import matplotlib.pyplot as plt
import librosa

# ma.random_playlist(5)
# ma.
actions = Actions()
# ma = Mixer_agent(tracklist=testqueue, actions=actions, queue=testqueue)
# ma.random_playlist(5)
#
# ma.load_playlist()
# print(ma.playlist)
# ma.reset()
# ma.generate_original_overlays(saveto=True)

# plt.plot(ma.audios[1].audio_frames[1])
# plt.show()
print(actions.adjust_pitch)