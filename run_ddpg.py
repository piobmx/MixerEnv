
from networks.Agent import DDPG
from networks.Evaluator import Evaluator
from mixer_agent import Mixer_agent
from Actions import Actions
from utilz import *
from tqdm import tqdm

EPOCHS = 2000

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

# The main training loop starts:
for e in range(EPOCHS):
    print(f"\nEPOCH: {e}")
    ma.C = 0
#     ddpg_agent.mixer.generate_new_overlays(wavname="ddpgnew", saveto=True)
    print(f"memory size: {len(ddpg_agent.memory)}")
    for i in tqdm(range(0, 500)):
        ddpg_agent.interact(e)
        if ddpg_agent.mixer.both_playing:
            ddpg_agent.train()
        ddpg_agent.n_episode += 1


    ddpg_agent.evaluator.update_epoch()
    ddpg_agent.evaluator.save_reward_tofile(e)
    if np.random.random() < 0.15:
        ddpg_agent.mixer.generate_new_overlays(wavname=f"ddpgnew{e}", saveto=True)
    ddpg_agent.mixer.reset()

