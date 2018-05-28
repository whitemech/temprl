import numpy as np

from rltg.agents.parameters.Parameter import AnnealedParameter
from rltg.agents.policies.Policy import Policy


class EGreedy(Policy):

    def __init__(self, epsilon=0.1, epsilon_start=None, decay_steps=10000, eval:bool=False):
        super().__init__(eval)
        if epsilon_start is None:
            epsilon_start = epsilon
        self.epsilon = AnnealedParameter(end=epsilon, start=epsilon_start, decay_steps=decay_steps)
        self.eval = eval


    def update(self, *args, **kwargs):
        self.epsilon.update()

    def choose_action(self, values=(), optimal=False, **kwargs):
        acts = np.arange(len(values))

        if not optimal and np.random.rand() < self.epsilon.get():
            action_idx = np.random.choice(acts)
        else:
            max_acts = acts[values == np.max(values)]
            # when multiple max actions, in training 'choice', in evaluation should be deterministic
            if self.eval:
                action_idx = max_acts[0]
            elif len(max_acts)==0:
                return 0
            else:
                action_idx = np.random.choice(max_acts)


        return action_idx
