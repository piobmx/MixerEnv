from networks.Agent import DDPG
from networks.Evaluator import Evaluator
from mixer_agent import Mixer_agent
from Actions import Actions
from utilz import *
from tqdm import tqdm

max_episodes = 500
EPOCHS = 20

actions = Actions()

ddpg_agent = DDPG(mixerEnv=None)
evltr = Evaluator(EPOCHS)
ma = Mixer_agent(tracklist=testqueue, actions=actions, queue=testqueue)
# ma.random_playlist(5)
ddpg_agent.mixer = ma
ddpg_agent.set_evaluator(evltr)
ma.load_playlist()
print(ma.playlist)
ma.set_rl_agent(ddpg_agent)
ma.reset()
ma.generate_original_overlays(saveto=True)
ma.C = 0
rp, rv, rp2, rv2 = ma.update_state()

epoch_rewards = []
for e in range(EPOCHS):
    print(f"EPOCH: {e}")
    ma.C = 0

    for i in tqdm(range(0, 500)):
    # while ddpg_agent.n_episode < max_episodes:
        # print(ddpg_agent.n_episode)
        ddpg_agent.interact(e)
        ddpg_agent.train()
        ddpg_agent.n_episode += 1

    ddpg_agent.mixer.reset()


print(ddpg_agent.evaluator.reward_history)
max_episodes = 500
EPOCHS = 20

actions = Actions()

ddpg_agent = DDPG(mixerEnv=None)
evltr = Evaluator(EPOCHS)
ma = Mixer_agent(tracklist=testqueue, actions=actions, queue=testqueue)
# ma.random_playlist(5)
ddpg_agent.mixer = ma
ddpg_agent.set_evaluator(evltr)
ma.load_playlist()
print(ma.playlist)
ma.set_rl_agent(ddpg_agent)
ma.reset()
ma.generate_original_overlays(saveto=True)
ma.C = 0
rp, rv, rp2, rv2 = ma.update_state()

epoch_rewards = []
for e in range(EPOCHS):
    print(f"EPOCH: {e}")
    ma.C = 0

    for i in tqdm(range(0, 500)):
    # while ddpg_agent.n_episode < max_episodes:
        # print(ddpg_agent.n_episode)
        ddpg_agent.interact(e)
        # if ddpg_agent.mixer.both_playing:
        ddpg_agent.train()
        ddpg_agent.n_episode += 1

    ddpg_agent.mixer.reset()