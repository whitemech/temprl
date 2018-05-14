from abc import abstractmethod


class RewardAutomaton():

    @abstractmethod
    def get_immediate_reward(self, q, q_prime, is_terminal_state=False):
        pass

    @abstractmethod
    def potential_function(self, q, is_terminal_state=False):
        pass
