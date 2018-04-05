import numpy as np
import matplotlib.pyplot as plt

class StatsManager(object):

    def __init__(self, window_size=300):
        self.total_reward_history = np.array([], dtype=np.int32)
        self.avg_reward_history = np.array([], dtype=np.float32)
        self.std_reward_history = np.array([], dtype=np.float32)
        self.explored_states_history = np.array([], dtype=np.int32)
        self.goals = np.array([], dtype=np.bool)


    def update(self, n_states, total_reward, goal):
        self.explored_states_history = np.append(self.explored_states_history, n_states)
        self.total_reward_history = np.append(self.total_reward_history, total_reward)
        avg_reward = np.mean(self.total_reward_history[-300:])
        std_reward = np.std(self.total_reward_history[-300:])
        self.avg_reward_history = np.append(self.avg_reward_history, avg_reward)
        self.std_reward_history = np.append(self.std_reward_history, std_reward)

        self.goals = np.append(self.goals, goal)


    def print_summary(self, episode_number, step, n_states, total_reward, epsilon, goal):
        if episode_number % 1 ==0:
            print('Episode: {}, Step: {:5d}, Explored States: {:7d}, Total Reward: {:6.2f}, AvgReward: {:05.2f}, StdReward: {:05.2f}, Epsilon: {:05.4f}, Goal: {}, GoalPerc: {:05.2f}'
              .format(episode_number, step, n_states, total_reward,
                      np.mean(self.total_reward_history[-300:]),
                      np.std(self.total_reward_history[-300:]),
                      epsilon, goal, np.mean(self.goals[-300:])*100)
                  )

    def plot(self):
        plt.plot(self.total_reward_history)
        plt.show()
        plt.plot(list(self.avg_reward_history))
        plt.show()
        plt.plot(list(self.std_reward_history))
        plt.show()
