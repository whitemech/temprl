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
        """Decides what to do as the next action"""
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args):
        """Called at the end of each episode"""
        raise NotImplementedError
