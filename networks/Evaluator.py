import numpy as np

class Evaluator():
    """
    For storing history of losses and rewards throughout the training process
    """
    def __init__(self, epochs_num):
        self.reward_history = [[] for _ in range(epochs_num)]
        self.losses_history = [[] for _ in range(epochs_num)]
        self.action_distribution = []
        self.times = [-1 for _ in range(epochs_num)]
        self.current_epoch = 0
        self.action_history = [[] for _ in range(epochs_num)]

    def print(self):
        pass

    def update_epoch(self):
        self.current_epoch += 1
        return

    def update_epoch_elapse_time(self, epoch, elapse_time):
        self.times[epoch] = elapse_time
        return

    def save_reward_tofile(self, epoch, dir=None):
        # np.savetxt(f'evaluation/rewards/reward_history{self.current_epoch}', self.reward_history, fmt='%1.4e')
        if dir is None:
            np.save(f'evaluation/rewards/reward_history', self.reward_history)
            np.save(f'evaluation/losses/losses_history', self.losses_history)
            np.save(f'evaluation/action/action_history', self.action_history)
        else:
            np.save(f'{dir}reward_history', self.reward_history)
            np.save(f'{dir}losses_history', self.losses_history)
            np.save(f'{dir}action_history', self.action_history)
        return