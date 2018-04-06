import random

from rltg.agents.exploration_policies.ExplorationPolicy import ExplorationPolicy


class RandomPolicy(ExplorationPolicy):

    def __init__(self, n_actions, epsilon_start=1.0, epsilon_end=0.01, decaying_steps=10000):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_steps = decaying_steps

        self.epsilon = self.epsilon_start
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

    def update(self, *args):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        else:
            self.epsilon = self.epsilon_end

    def explore(self, *args):
        action_id = None
        if random.random() < self.epsilon:
            action_id = random.randrange(self.n_actions.n)
        return action_id

    def reset(self, *args):
        pass
