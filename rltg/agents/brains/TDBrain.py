import random
from abc import abstractmethod

import numpy as np
import math

from gym.core import Space
from gym.spaces import Discrete

from rltg.agents.brains.Brain import Brain


class TDBrain(Brain):
    def __init__(self, observation_space:Discrete, action_space:Space, gamma=0.99, alpha=None, nsteps=300):
        super().__init__(observation_space, action_space)

        self.Visits = {}
        self.gamma = gamma
        self.alpha = alpha

        # sparse representation
        self.Q = {}
        self.nsteps = nsteps
        self.obs_history = []

    def choose_action(self, state, optimal=False):
        Q_values = self.Q[state] if state in self.Q else np.zeros((self.action_space.n,))
        argmaxes = np.argwhere(Q_values == Q_values.max()).flatten()
        if optimal:
            # determistic behavior!
            action = argmaxes[0]
        else:
            # allow randomness
            action = random.choice(argmaxes)

        # action = random.choice(argmaxes)
        return action

    def learn(self):
        if self.episode_iteration + 1 < self.nsteps:
            # no enough observations. + 1 for manage the case when nsteps=1 and iteration=0
            return

        tau = self.episode_iteration + 1 - self.nsteps
        self._nsteps_update(self.obs_history[tau: tau + self.nsteps])

    def observe(self, state, action, reward, state2):
        super().observe(state, action, reward, state2)
        # self.obs_history.append((state, action, reward, state2))
        # if len(self.obs_history)==self.nsteps:
            # delete oldest element
            # self.obs_history.pop(0)
        self.incVisits(state, action)

    def reset(self):
        t = self.episode_iteration - self.nsteps
        for i in range(t, self.episode_iteration):
            self._nsteps_update(self.obs_history[i:], last=True)

        self.obs_history = []
        super().reset()


    def _nsteps_update(self, partial_history, last=False):
        if len(partial_history)<1:
            return

        n_reward_return = 0
        N = len(partial_history)
        # gamma^0*r0 + gamma^1*r1 + ... + gamma^N-1*r_n-1
        for i in range(N):
            n_reward_return += partial_history[i][2] * math.pow(self.gamma, i)

        if not last:
            # add the value of the Q function, depending on the type of algorithm (see the getQa method)
            _, _, _, s_tn = partial_history[-1]
            Q_tn = self.getQa(s_tn)
            n_reward_return += math.pow(self.gamma, N) * Q_tn

        s_tau, a_tau, _, _ = partial_history[0]
        delta = n_reward_return - self.Q.get(s_tau, np.zeros(self.action_space.n))[a_tau]

        if s_tau not in self.Q:
            self.Q[s_tau] = np.zeros(self.action_space.n)
        self.Q[s_tau][a_tau] += self.getAlphaVisitsInc(s_tau, a_tau) * delta

    def setVisits(self, x, a, q):
        if not x in self.Visits:
            self.Visits[x] = np.zeros(self.action_space.n)
        self.Visits[x][a] = q

    def incVisits(self, x, a):
        self.setVisits(x,a,self.getVisits(x,a)+1)

    def getVisits(self, x, a):
        if x in self.Visits:
            return self.Visits[x][a]
        else:
            return 0

    def getAlphaVisitsInc(self, x, a):
        if self.alpha:
            return self.alpha

        self.incVisits(x,a)
        s = self.getVisits(x,a)
        try: #TODO debug here
            a = 1.0/float(s)
        except:
            a = 1.0
        #print("visits: %d, a = %.6f" %(s,a))
        return a # math.sqrt(s)

    def getSumVisits(self, x):
        return np.sum(self.Visits[x,:])


    @abstractmethod
    def getQa(self, s):
        raise NotImplementedError



class QLearning(TDBrain):
    def __init__(self, observation_space:Discrete, action_space, gamma=0.99, alpha=None, nsteps=200):
        super().__init__(observation_space, action_space, gamma, alpha, nsteps)

    def getQa(self, s):
        maxQa = max(self.Q.get(s, np.zeros(self.action_space.n)))
        return maxQa


class Sarsa(TDBrain):
    def __init__(self, observation_space:Discrete, action_space, gamma=0.99, alpha=None, nsteps=200):
        super().__init__(observation_space, action_space, gamma, alpha, nsteps)

    def getQa(self, s):
        a = self.choose_action(s)
        q = self.Q.get(s, np.zeros(self.action_space.n))[a]
        return q
