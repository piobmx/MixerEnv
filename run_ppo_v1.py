import gym
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from networks.Evaluator import Evaluator
from mixer_agent import Mixer_agent
from utilz import *
from tqdm import tqdm
from networks.PPO import PPO


###
# Implementation of the PPO Pitch-matching algorithm described in the report
###

has_continuous_action_space = True  # continuous action space; else discrete
evaluator = Evaluator(2000)
ma = Mixer_agent(tracklist=testqueue, queue=testqueue)
ma.load_playlist()
ma.reset()
ma.reset_onsets()
# ma.nn_size = 50
input_dim, action_dim = ma.reset_pitch_by_frame()

train_gym = False
max_ep_len = 1500 # max timesteps in one episode
# max_training_timesteps = int(4e6)   # break training loop if timeteps > max_training_timesteps
max_training_timesteps = int(1.5e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.01        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.05                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = 1e4  # action_std decay frequency (in num timesteps)
update_timestep = 2e3      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update
hidden_size_1 = 256//4
hidden_size_2 = 256//4

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.8            # discount factor

lr_actor = 0.001       # learning rate for actor network
lr_critic = 0.01       # learning rate for critic network
data = {
    "action_std": action_std,
    "action_std_decay_rate": action_std_decay_rate,
    "min_action_std": min_action_std,
    'action_std_decay_freq': action_std_decay_freq,
    'K_epochs': K_epochs,
    "eps_clip": eps_clip,
    "gamma": gamma,
    "lr_actor": lr_actor,
    "lr_critic": lr_critic,
    "max_training_timesteps": max_training_timesteps,
}

random_seed = 0         # set random seed if required (0 = no random seed)
env_name = "Pendulum-v0"
print("training environment name : " + env_name)
env = gym.make(env_name)

# state space dimension
state_dim = input_dim
now = nowness()
eval_dir = f"evaluation/{now}/"
#### get number of log files in log directory
run_num = 0
# current_num_files = next(os.walk(eval_dir))[2]
# run_num = len(current_num_files)


#### create new log file for each run
log_f_name = eval_dir + '/PPO_' + env_name + "_log_" + "log_f_name"+ ".csv"
print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)
################### checkpointing ###################
run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder
directory = "PPO_preTrained"

# with open(f'{eval_dir}data.p', 'wb') as fp:
#     pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
print(f"STATE DIM: {state_dim}, ACTION_DIM: {action_dim}")
ppo_agent = PPO(state_dim=state_dim, action_dim=action_dim, actor_lr=lr_actor, critic_lr=lr_critic, gamma=gamma,
                K_epochs=K_epochs, eps_clip=eps_clip, has_continuous_action_space=True, action_std_init=action_std,
                hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2)

os.makedirs(eval_dir, exist_ok=True)
# write_agent_info(ppo_agent, path=eval_dir)
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)
with open(f'{eval_dir}data.json', 'w') as fp:
    json.dump(data, fp)
fp.close()
log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')
print_running_reward = 0
print_running_episodes = 0
log_running_reward = 0
log_running_episodes = 0
time_step = 0
i_episode = 0
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)

total_losses = []
total_reward = []
total_distan = []
total_action = []
total_step_reward = []

use_sox = True
update_live = False
while time_step <= max_training_timesteps:

    current_ep_reward = 0
    current_ep_steps_reward = []
    observation_pre = ma.default_state
    # observation_pre = env.reset()
    render_flag = False
    e_loss = 0
    ma.episode = 0
    if i_episode > 60 and np.random.random() < 0.2 and train_gym:
        render_flag = True
    epoch_dis = []
    # for t in tqdm(range(1, max_ep_len+1)):
    for t in tqdm(range(1, ma.MAX_C)):
        if render_flag and train_gym: env.render()
        # select action with policy
        action = ppo_agent.select_action(observation_pre)

        observation_post, reward, done, distance = \
            ma.live_pitch_match(action, use_sox=use_sox, update_live=update_live)
        epoch_dis.append(distance)

        ppo_agent.buffer.rewards.append(reward)
        ma.update_C()
        observation_pre = ma.get_next_pre_state()
        # ppo_agent.buffer.is_terminals.append(done)
        time_step +=1
        ma.episode += 1
        current_ep_reward += reward
        current_ep_steps_reward.append(reward)
        # update PPO agent
        if time_step % update_timestep == 0:
            loss = ppo_agent.update()
            e_loss += loss
            total_losses.append(e_loss)
        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
        # log in logging file
        if time_step % log_freq == 0 or time_step == max_training_timesteps:
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)
            print(f"EPS: {ppo_agent.action_std}")
            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()
            log_running_reward = 0
            log_running_episodes = 0
            print("maC: ", ma.C)
            if ma.C > 130:
                ppo_agent.save(eval_dir + f"{time_step}_{checkpoint_path}")
                ma.plot_pitches(eval_dir, step=i_episode)
                np.save(eval_dir + f"best_dist_log{time_step}.npy", ma.best_dist_log)
                np.save(eval_dir + f"best_action_log{time_step}.npy", ma.best_action_log)
                np.save(eval_dir + f"new_state{time_step}.npy", ma.best_log)
                if np.random.random() < 0.98 and use_sox:
                    np.save(f"{eval_dir}{i_episode}_modulated.npy", ma.best_modulated)

        # printing average reward
        if time_step % print_freq == 0:
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
            print_running_reward = 0
            print_running_episodes = 0
        # save model weights
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            # print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
        # break; if the episode is over
        if done:
            break
    if e_loss != 0: print(f"episode loss: {e_loss} at i_episode: {i_episode}")
    total_reward.append(current_ep_reward)
    total_step_reward.append(current_ep_steps_reward)
    total_distan.append(epoch_dis)
    total_action.append(ma.epoch_action_log)
    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1
    # ma.plot_explored_action_space(eval_dir, step=time_step)
    # ppo_agent.action_std = action_std
    ma.episodic_reset_pitch_by_frame()

    if i_episode % 12 == 1 or i_episode < 12 or time_step == max_training_timesteps:
        np.save(eval_dir+"total_reward.npy", total_reward)
        np.save(eval_dir+"total_losses.npy", total_losses)
        np.save(eval_dir+"total_distance.npy", total_distan)
        np.save(eval_dir+"total_step_reward.npy", total_step_reward)
        np.save(eval_dir+"total_action.npy", total_action)
        fig, ax = plt.subplots(3, 1, figsize=(20, 10), dpi=120)
        ax[0].plot(total_reward)
        ax[0].set_title("REWARD OVER EPOCHS")
        ax[1].plot(total_losses)
        ax[1].set_title("ACTOR-CRITIC LOSS")
        ax[2].plot(epoch_dis, linewidth=1, color="red", label="modulated")
        ax[2].plot(ma.distance, linewidth=0.5, color="black", label="original")
        ax[2].set_title("Distances")
        ax[2].legend()
        plt.savefig(eval_dir+"mid.png")
        plt.close()
    i_episode += 1
log_f.close()