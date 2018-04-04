import numpy as np
import math

from gym.core import Space

from rltg.brains.Brain import Brain


class TDBrain(Brain):
    def __init__(self, state_space:Space, action_space:Space, gamma=0.99, nsteps=300):
        super().__init__(state_space, action_space)

        self.Visits = {}
        self.gamma = gamma

        # sparse representation
        self.Q = {}

        self.episode = 0
        self.iteration = 0
        self.episode_iteration = 0
        self.nsteps = nsteps
        self.obs_history = []


    def observe(self, state, action, reward, state2):
        self.obs_history.append((state, action, reward, state2))
        self.incVisits(state, action)
        self.iteration += 1
        self.episode_iteration += 1

    def reset(self):
        t = self.episode_iteration - self.nsteps
        for i in range(t, self.episode_iteration):
            self._nsteps_update(self.obs_history[i:])

        self.obs_history = []
        self.episode += 1
        self.episode_iteration = 0

    def _nsteps_update(self, partial_history):
        n_reward_return = 0
        N = len(partial_history)
        for i in range(N):
            n_reward_return += partial_history[i][2] * math.pow(self.gamma, i)

        s_tn, a_tn, _, _ = partial_history[-1]
        Q_tn = self.Q.get(s_tn, np.zeros(self.action_space.n))[a_tn]
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


class QLearning(TDBrain):
    def __init__(self, state_space, action_space, gamma=0.99, nsteps=200):
        super().__init__(state_space, action_space, gamma, nsteps)


    def best_action(self, state):
        Q_values = self.Q[state] if state in self.Q else np.zeros((self.action_space.n,))
        action = np.argmax(Q_values)
        return action

    def learn(self):
        if self.episode_iteration + 1 < self.nsteps:
            return

        tau = self.episode_iteration - self.nsteps
        self._nsteps_update(self.obs_history[tau: tau + self.nsteps])

class Sarsa(TDBrain):
    def __init__(self, state_space, action_space, gamma=0.99, nsteps=200):
        super().__init__(state_space, action_space, gamma, nsteps)
        # initialize at any value
        self.next_action = 0


    def best_action(self, state):
        action = self.next_action
        return action

    def learn(self):
        if self.episode_iteration + 1 < self.nsteps:
            return

        tau = self.episode_iteration - self.nsteps
        self._nsteps_update(self.obs_history[tau: tau + self.nsteps])


    def observe(self, state, action, reward, state2):
        super().observe(state, action, reward, state2)
        s_ = state2
        next_Q = self.Q[s_] if s_ in self.Q else np.zeros(self.action_space.n)
        self.next_action = np.argmax(next_Q)
