from abc import ABC, abstractmethod


class ExplorationPolicy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def update(self, *args):
        """Called at each episode iteration"""
        raise NotImplementedError

    @abstractmethod
    def explore(self, *args):
        """Decides what to do as the next action. The actual implementation must be defined into the "_explore" method.
        :returns the action to pick or None if the exploration policy decides to don't explore."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args):
        """Called at the end of each episode"""
        raise NotImplementedError
