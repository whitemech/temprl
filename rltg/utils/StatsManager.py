import pickle

import numpy as np
import matplotlib.pyplot as plt

class StatsManager(object):

    def __init__(self, name="stats", window_size=100):
        self.name = name
        self.window_size = window_size
        self.steps_taken= np.array([], dtype=np.int32)
        self.total_reward_history = np.array([], dtype=np.int32)
        self.avg_reward_history = np.array([], dtype=np.float32)
        self.std_reward_history = np.array([], dtype=np.float32)
        self.explored_states_history = np.array([], dtype=np.int32)
        self.goals = np.array([], dtype=np.bool)


    def update(self, steps, n_states, total_reward, goal):
        self.steps_taken = np.append(self.steps_taken, steps)
        self.explored_states_history = np.append(self.explored_states_history, n_states)
        self.total_reward_history = np.append(self.total_reward_history, total_reward)
        avg_reward = np.mean(self.total_reward_history[-self.window_size:])
        std_reward = np.std(self.total_reward_history[-self.window_size:])
        self.avg_reward_history = np.append(self.avg_reward_history, avg_reward)
        self.std_reward_history = np.append(self.std_reward_history, std_reward)

        self.goals = np.append(self.goals, goal)


    def print_summary(self, episode_number, step, n_states, total_reward, epsilon, goal):
        w = self.window_size
        if episode_number % 1 ==0:
            print('Episode: {:6d}, Step: {:5d}, Explored States: {:7d}, Total Reward: {:8.2f}, AvgReward: {:8.2f}, StdReward: {:8.2f}, Epsilon: {:05.4f}, Goal: {:5}, GoalPerc: {:05.2f}'
              .format(episode_number, step, n_states, total_reward,
                      np.mean(self.total_reward_history[-w:]),
                      np.std(self.total_reward_history[-w:]),
                      epsilon, str(goal), np.mean(self.goals[-w:])*100)
                  )

    def plot(self):
        plt.figure(1)
        plt.title("Total reward history")
        plt.plot(self.total_reward_history)
        plt.figure(2)
        plt.title("Average total reward history")
        plt.plot(list(self.avg_reward_history))
        plt.figure(3)
        plt.title("Standard deviation total reward history")
        plt.plot(list(self.std_reward_history))
        plt.show()

    def to_csv(self, filepath):
        with open(filepath + ".csv", "w") as f:
            f.write("Episode;Steps;Total reward;Explored states;Goal\n")
            for ep, (steps, total_rewards, explored_states, goals) in enumerate(zip(
                self.steps_taken, self.total_reward_history, self.explored_states_history, self.goals
            )):
                f.write(";".join(map(str,[ep, steps, total_rewards, explored_states, goals])) + "\n")
