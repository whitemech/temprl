import random

from gym import Space

from rltg.agents.exploration_policies.ExplorationPolicy import ExplorationPolicy


class RandomPolicy(ExplorationPolicy):

    def __init__(self, action_space:Space, epsilon=0.1, epsilon_start=None, decaying_steps=1):
        super().__init__()
        self.action_space = action_space
        self.exploration_steps = decaying_steps

        if epsilon_start is None:
            self.epsilon = epsilon
            self.epsilon_start = None
        else:
            self.epsilon_start = epsilon_start
            self.epsilon_end = epsilon
            self.epsilon = self.epsilon_start
            self.epsilon_decay_step = (epsilon_start - epsilon) / self.exploration_steps

    def update(self, *args):
        if self.epsilon_start is not None:
            if self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_decay_step
            else:
                self.epsilon = self.epsilon_end

    def explore(self, *args):
        action_id = None
        if random.random() < self.epsilon:
            action_id =self.action_space.sample()
        return action_id

    def reset(self, *args):
        pass
