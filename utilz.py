import datetime, pickle, json, os
from pathlib import Path

import numpy as np
import librosa
import crepe
# from statistics import mode, multimode
from sklearn.preprocessing import normalize

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
"test_folder/for_main/We Have Explosive.wav",
"test_folder/for_main/Implosive.wav",

]


def audio_to_spec(time_series):
    melspec = librosa.feature.melspectrogram(time_series, sr=global_sr, hop_length=512, fmax=8000)
    #     D = librosa.power_to_db(melspec, ref=np.max)
    lpd = librosa.power_to_db(melspec, ref=np.max)

    return lpd[::-1]

def pitch_features(segment, hop_length=1024, nor=True, to_midi=False):
    if nor is True and to_midi is True:
        to_midi = not to_midi

    hop_length = hop_length
    p_features = np.array([])
    pitches, magnitudes = librosa.piptrack(y=segment, sr=global_sr, fmin=20, fmax=8000,
                                           n_fft=hop_length * 2, hop_length=hop_length)
    p = [pitches[magnitudes[:, i].argmax(), i] for i in range(0, pitches.shape[1])]
    pitch0 = np.array(p)  # shape (305,)
    pitch = np.transpose(pitch0)
    p_features = np.hstack((p_features, max(20, np.amin(pitch, 0))))
    p_features = np.hstack((p_features, np.amax(pitch, 0)))
    p_features = np.hstack((p_features, np.median(pitch, 0)))
    p_features = np.hstack((p_features, np.mean(pitch, 0)))
    p_features = np.hstack((p_features, np.std(pitch, 0)))
    # p_features = np.hstack((p_features, np.var(pitch, 0)))
    if nor:
        p_features = normalize(p_features.reshape(1, -1))
    if to_midi:
        p_features = np.int_(librosa.hz_to_midi(p_features))
    return p_features

def get_pitches_by_beats(y, starts, ends):
    pitches = []
    for b in range(len(starts)):
        seg1 = y[np.int(starts[b] * global_sr): np.int(ends[b] * global_sr)]
        pitches1, magnitudes1 = librosa.piptrack(y=seg1, sr=global_sr, fmin=20, fmax=8000, hop_length=2048)
        max_p = np.median((extract_max(pitches1)))
        pitches.append(max_p)
    return np.array(pitches)

def extract_max(pitches):
    new_pitches = []
    for i in range(0, pitches.shape[1]):
        new_pitches.append(np.max(pitches[:, i]))
    return new_pitches

def estimate_pitch(segment, sr=44100):
    time, frequency, confidence, activation = crepe.predict(segment,
                                                            sr=sr,
                                                            viterbi=True,
                                                            model_capacity='tiny',
                                                            verbose=0
                                                            )
    return frequency

def estimate_pitch_ac(segment, sr=global_sr, fmin=50.0, fmax=2000.0):
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

def delete_empty_wav(path):
    for x in path.iterdir():
        if os.stat(str(x)).st_size < 10**4:
            os.remove(x)
    return 0

def nowness():
    now = datetime.datetime.now()
    timestamp = f"{str(now.month)}{str(now.day)}_h{str(now.hour)}m{now.minute}s{now.second}"
    return timestamp

def join_frames(frames):
    nframes = np.array(frames)
    joined = np.concatenate(nframes)
    if nframes.size == joined.size:
        return joined
    else:
        return None

def write_ppo_agent_info(path):
    return 0

def write_agent_info(rlagent, path, write_complete=False):
    info = rlagent.__dict__

    if write_complete:
        with open(f'{path}data.p', 'wb') as fp:
            pickle.dump(info, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        data = {
            "gamma": info['gamma'],
            "target_tau": info["target_tau"],
            "actor_lr": info['actor_lr'],
            'critic_lr': info['critic_lr'],
            'actor_optimizer': info['actor_optimizer'].__dict__['_zero_grad_profile_name']
        }
        with open(f'{path}data.p', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{path}data.json', 'w') as fp:
            json.dump(data, fp)
        fp.close()

def soxx(sox_transformer, audio):
    modulated = sox_transformer.build_array(input_array=audio.copy(), sample_rate_in=global_sr)
    return modulated

def obs_matrix(matrix, frame_matrix, default_length=550, default_num=5):
    obs = np.zeros((default_length,))
    for i in range(default_num - 1):
        difx = np.abs(frame_matrix[i] + frame_matrix[i + 1])
        difx[np.where(difx != 2)] = 0
        difx[np.where(difx == 2)] = 1
        dif = np.abs((matrix[i] - matrix[i + 1])) * difx
        selecting = np.where(dif > 24)
        obs += dif
    return obs


def detect_onsets(y, lag=2):
    onset_env = librosa.onset.onset_strength(
        y, sr=global_sr,
        aggregate=np.median,
        lag=lag
    )
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=global_sr)
    times = librosa.times_like(onset_env, sr=global_sr)
    return onset_env, beats, times

def find_nearest(beats1, beats2, forward=False, step=False):
    """
    For each beat in list beats2, find nearest beat in beats1 regarding time
    """
    # assert beats1[0] < beats2[0] and beats1[-1] < beats2[-1]
    dis_arr = np.zeros(beats2.shape)
    dis_ind = np.zeros(beats2.shape)
    x = 1 + 1 * step
    for index, b2 in enumerate(beats2[1::x]):
        if forward:
            if b2 > beats1.max():
                min_dis = 9
            else:
                distance = beats1[1::x] - b2
                distance *= distance > 0
                min_dis = distance[np.argmax(distance > 0)]
                min_dis = np.clip(min_dis, 0, 10)
        if not forward:
            adistance = beats1[1::x] - b2
            distance = np.abs(beats1[1::x] - b2)
            indx = np.argmin(distance)
            # min_dis = np.min(distance)
            min_dis = adistance[indx]
            ina = np.argmin(distance) + 1
            dis_ind[index + 1] = int(ina)
        dis_arr[index + 1] = min_dis
    return dis_arr, dis_ind


def distance_quality(mean):
    return 1 / (np.log(mean + 0.0001) - 1)

def re_combine(ta, tb, ind):
    if tb.size > ta.size:
        new = np.zeros(tb.shape)
    else:
        new = np.zeros(ta.shape)
    new[:ind] = ta[:ind]
    new[ind:len(tb)] = tb[ind:]
    return new

def seperate(y, mark, bts):
    ind = np.int_(bts[mark] * global_sr)
    p2, p1 = y.copy(), y.copy()
    p1[ind:] = 0
    p2 = p2[ind:]
    return p1, p2, ind

def l2_reward(new_state, ref_state):
    max_index_1 = np.argmax(new_state)
    max_index_2 = np.argmax(ref_state)
    distance = np.linalg.norm(new_state - ref_state)
    if max_index_2 == max_index_1 or np.abs(max_index_2 - max_index_1) == 5 \
            or np.abs(max_index_2 - max_index_1) == 7:
        env_reward = 1
    else:
        env_reward = -distance
        env_reward = np.clip(env_reward, -1, 1)
    return env_reward