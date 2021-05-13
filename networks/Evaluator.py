import numpy as np

class Evaluator():
    """
    For storing history of losses and rewards throughout the training process
    """

    def __init__(self, epochs_num):
        self.reward_history = [[] for _ in range(epochs_num)]
        self.losses_history = [[] for _ in range(epochs_num)]
        self.action_distribution = []
        self.current_epoch = 0

    def print(self):
        pass

    def update_epoch(self):
        self.current_epoch += 1
        return

    def save_reward_tofile(self, epoch):
        # np.savetxt(f'evaluation/rewards/reward_history{self.current_epoch}', self.reward_history, fmt='%1.4e')
        np.save(f'evaluation/rewards/reward_history{self.current_epoch}', self.reward_history)
        np.save(f'evaluation/losses/losses_history{self.current_epoch}', self.losses_history)
        return 1