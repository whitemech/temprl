import functools
import operator
import typing
from abc import ABC, abstractmethod

from gym import Space
from gym.spaces import Tuple, Discrete


class FeatureExtractor(ABC):

    def __init__(self, input_space: Space, output_space: Space):
        self.input_space = input_space
        self.output_space = output_space

    def __call__(self, input, **kwargs):
        """Extract features from the input and make sanity check
        of the input and the output dimensions of the abstract method '_extract'"""

        if not self.input_space.contains(input):
            raise ValueError("input space dimensions are not correct.")

        output = self._extract(input, **kwargs)

        if not self.output_space.contains(output):
            raise ValueError("output space dimensions are not correct.")

        return output

    @abstractmethod
    def _extract(self, input, **kwargs):
        """
        Extract the feature from `input` (contained into `self.input_space`).
        The features must be contained into `self.output_space`.
        :param   input:  a state belonging to input_space
        :returns output: a state belonging to output_space
        """
        raise NotImplementedError


class RobotFeatureExtractor(FeatureExtractor):
    """A specialization of the FeatureExtractor used from RLAgent."""
    pass



class IdentityFeatureExtractor(FeatureExtractor):
    def __init__(self, space:Space):
        super().__init__(space, space)

    def _extract(self, input, **kwargs):
        return input


class TupleFeatureExtractor(FeatureExtractor):
    """Collapse a tuple state to an integer, according to the ranges of each dimensions.
    For example:
    consider the scenario with this observation space:
    >>> tuple_space = Tuple((Discrete(10), Discrete(5), Discrete(20)))
    >>> extractor = TupleFeatureExtractor(tuple_space)

    The state (9, 3, 11) which is contained in the observation space, is converted in the following way:
    >>> extracted_feature = (11*10 + 9)*5+3
    >>> extracted_feature
    598
    >>> extractor((9, 3, 11))
    598

    The maximum state (9, 4, 19) is represented in this transformed space as:
    >>> extracted_feature = (19*10 + 9)*5+4
    >>> extracted_feature
    999
    >>> extractor((9, 4, 19))
    999

    which is equal to 20*10*5 - 1

    we sort the space components in descending order of their dimensions, otherwise this method wouldn't work.
    """

    def __init__(self, space:Tuple):
        # assume every subspace is of type Discrete
        # TODO: generalize to different subspaces?
        assert isinstance(space, Tuple) and all(isinstance(s, Discrete) for s in space.spaces)

        # compute output dimension by multiplying all the dimensions
        # i.e.: given a space (N0, N1, N2 ... Nn)
        # the total dimension is: N0*N1*...*Nn
        tot_dim = functools.reduce(operator.mul, map(lambda x: x.n, space.spaces))

        # sort subspaces from the biggest to the smaller, keeping its component id in the input space
        self.id2space_sorted = sorted(enumerate(space.spaces), key=lambda x: -x[1].n)

        super().__init__(space, Discrete(tot_dim))


    def _extract(self, input, **kwargs):

        # new_input = [input[id] for id, _ in self.id2space_sorted]

        # sort the input by descending size of the space components
        state = input[self.id2space_sorted[0][0]]

        # collapse the input into only one number,
        # for doing this, we need to start
        # from the biggest space to the smallest.
        for id, space in self.id2space_sorted[1:]:
            state *= space.n
            state += input[id]

        return state

