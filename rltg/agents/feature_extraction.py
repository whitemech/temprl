import typing
from abc import ABC, abstractmethod

from gym import Space
from gym.spaces import Tuple, Discrete


class FeatureExtractor(ABC):

    def __init__(self, input_space: Space, output_space: Space):
        self.input_space = input_space
        self.output_space = output_space

    def __call__(self, input, **kwargs):
        if not self.input_space.contains(input):
            raise ValueError("input space dimensions are not correct.")

        output = self._extract(input, **kwargs)

        if not self.output_space.contains(output):
            raise ValueError("output space dimensions are not correct.")

        return output

    @abstractmethod
    def _extract(self, input, **kwargs):
        """
        :param   input:  a state belonging to input_space
        :returns output: a state belonging to output_space
        """
        raise NotImplementedError


class RobotFeatureExtractor(FeatureExtractor):

    class RobotState(object):
        """An helper state wrapper to make RobotFeatureExtractor more modular
        Actually this class is used only in the implementation of the TGAgent
        """
        def __init__(self, world_state, automata_states: typing.Tuple):
            self.world_state = world_state
            self.automata_states = automata_states

    def __init__(self, input_space: Space, output_space: Space):
        super().__init__(input_space, output_space)

    def __call__(self, input, **kwargs):
        """
        :param   input: world_state or an instance of RobotState (only when called by a TGAgent)
        :return: the output of the abstract method "_extract"
        """
        if isinstance(input, self.RobotState):
            return super().__call__(input.world_state, automata_states=input.automata_states)
        else:
            return super().__call__(input)

    @abstractmethod
    def _extract(self, input, automata_states: typing.Tuple=None):
        """
        :param   input:            a state belonging to input_space
        :param   automata_states:  tuple of ids of automata states.
                                   It should be considered only if the feature extractor
                                   is destined to a Temporal Goal task.
        :returns output:           a state belonging to output_space
        """
        raise NotImplementedError



class IdentityFeatureExtractor(FeatureExtractor):
    def __init__(self, space:Space):
        super().__init__(space, space)

    def _extract(self, input, **kwargs):
        return input


class TupleFeatureExtractor(FeatureExtractor):
    """Collapse a tuple state to an integer, according to the ranges of each dimensions.
    For example:
    consider the scenario with this observation space: Tuple((Discrete(10), Discrete(20))
    The state (5, 11) which is contained in the observation space, is converted in the following way:
    >>>extracted_feature = (11*20 + 5)*10
    we start from the biggest space component, otherwise this method wouldn't work.
    """

    def __init__(self, space:Tuple):
        # assume every subspace is of type Discrete
        # TODO: generalize to different subspaces?
        assert isinstance(space, Tuple) and all(isinstance(s, Discrete) for s in space.spaces)

        # compute output dimension by computing the max id state
        dim = 1
        for s in space.spaces:
            dim *= s.n * (s.n-1)

        super().__init__(space, Discrete(dim))

        # sort subspaces from the biggest to the smaller
        self.id2space_sorted = sorted(enumerate(space.spaces), key=lambda x: -x[1].n)


    def _extract(self, input, **kwargs):
        state = 0
        # collapse the input into only one number,
        # for doing this, we need to start
        # from the biggest space to the lowest.
        for id, space in self.id2space_sorted:
            state += input[id]
            state *= space.n

        # just add the last dimension component
        # state += input[-1]

        return state
