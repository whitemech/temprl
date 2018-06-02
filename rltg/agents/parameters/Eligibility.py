from collections import defaultdict

from rltg.agents.parameters.Parameter import Parameter
from rltg.utils.misc import mydefaultdict


class Eligibility(Parameter):

    def __init__(self, lambda_, gamma):
        super().__init__()
        self.lambda_ = lambda_
        self.gamma = gamma

        self.traces = mydefaultdict(0.0)

    def get(self, state, action):
        return self.traces[(state, action)]

    def to_zero(self, state, action):
        self.pop(state, action)

    def to_one(self, state, action):
        self.reset_state_action(state, action)

    def reset_state_action(self, state, action):
        self.traces[(state, action)] = 1

    def pop(self, state, action):
        self.traces.pop((state, action))

    def update(self, state, action, *args, **kwargs):
        self.traces[(state, action)] = self.gamma * self.lambda_ * self.traces[(state, action)]
        if self.traces[(state, action)] < 1e-4:
            self.pop(state, action)

    def reset(self):
        self.traces = {}
