from abc import abstractmethod


class RewardAutomaton():

    @abstractmethod
    def potential_function(self, q, is_terminal_state=False):
        raise NotImplementedError
