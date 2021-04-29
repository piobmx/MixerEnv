import numpy as np
import librosa
from pathlib import Path

global_sr = 44100 # global sample rate
global_hop_len = 4410 # 10 frames per seconds
frame_len = global_hop_len
testwav = "test_folder/wavs/Diversion_p0.wav"
wav_folder = "test_folder/wavs"
graph_folder = "graphs/"
wav_path = Path(wav_folder)
# testqueue = ['test_folder/wavs/One Minute To Midnight_p0.wav',
#              'test_folder/wavs/Genesis_p0.wav',
#              'test_folder/wavs/Three Signs From The Other Side (Second Sign)_p0.wav',
#              'test_folder/wavs/Three Signs From The Other Side (Third Sign)_p0.wav',
#              'test_folder/wavs/Dancing Shadows_p1.wav']

testqueue = [
"candidates/Punk To Funk_p1.wav",
"candidates/Porcelain_p1.wav",
"candidates/Chemical Beats_p1.wav",
"candidates/Natural Blues_p1.wav",
"candidates/10th & Crenshaw_p1.wav"
]

def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(segment)

    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr / fmax
    i_max = sr / fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0

    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr) / i
    return f0


def generate_sine(f0, sr, n_duration):
    n = np.arange(n_duration)
    return 0.2*np.sin(2*np.pi*f0*n/float(sr))

# gs = generate_sine(441, sr=global_sr, n_duration=10000)
# print(estimate_pitch(gs, sr=global_sr))


def delete_empty_wav(path):
    for x in path.iterdir():
        if os.stat(str(x)).st_size < 10**4:
            os.remove(x)
    return 0


def pitch_only(input, pitch_change=1):
    mix_ed = np.array([])
    # mix_ed = AudioSegment.empty()
    testclip0 = y[:100000]
    testclip = testclip0.copy()
    length_change = 1.9
    speed_fac = 1.0 / length_change
    print("resample length_change = ", length_change)
    tmp = np.interp(np.arange(0, len(testclip), speed_fac),
                    np.arange(0, len(testclip)), testclip)
    minlen = min(testclip.shape[0], tmp.shape[0])
    testclip *= 0
    testclip[0:minlen] = tmp[0:minlen]
    result = np.split(testclip, [minlen])[0]

    new_duration = librosa.get_duration(y=result, sr=global_sr)
    #     print(f"input audio duration: {duration}\noutput audio duration: {new_duration}")

    return result



def nowness():
    now = datetime.datetime.now()
    timestamp = f"{str(now.month)}{str(now.day)}_{str(now.hour)}{now.minute}{now.second}"
    return timestamp



def batch_tempo_alys(wav_file_list, write=False):
    for wfile in wav_file_list:
        if ".wav" not in str(wfile):
            continue
        print(str(wfile))
        tempo = vanilla_tempo_estimation(str(wfile))
        print(tempo)
#         print(str(wfile))

        if write:
            txtfile = str(wfile).replace(".wav", ".txt")
            with open(txtfile, "w") as tfile:
                tfile.write(f"tempo: {str(tempo[0])}")
    return 1


def vanilla_tempo_estimation(input_file):
    y, sr = librosa.load(input_file)
    onset_env = librosa.onset.onset_strength(y, sr=global_sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return tempo


def join_frames(frames):
    nframes = np.array(frames)
    joined = np.concatenate(nframes)
    if nframes.size == joined.size:
        return joined
    else:
        return None
