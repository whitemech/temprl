from abc import ABC, abstractmethod

from gym import Space
from gym.spaces import Tuple, Discrete


class FeatureExtractor(ABC):

    def __init__(self, input_space:Space, output_space:Space):
        self.input_space = input_space
        self.output_space = output_space

    def __call__(self, input):
        if not self.input_space.contains(input):
            print(self.input_space)
            print(input)
            print(self.__class__)
            self.input_space.contains(input)
            raise ValueError("input space dimensions are not correct.")

        output = self._extract(input)

        if not self.output_space.contains(output):
            raise ValueError("output space dimensions are not correct.")

        return output

    @abstractmethod
    def _extract(self, input):
        raise NotImplementedError


class IdentityFeatureExtractor(FeatureExtractor):
    def __init__(self, space:Space):
        super().__init__(space, space)

    def _extract(self, input):
        return input


class TupleFeatureExtractor(FeatureExtractor):
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


    def _extract(self, input):
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
