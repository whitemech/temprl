from abc import ABC, abstractmethod

from gym import Space


class FeatureExtractor(ABC):

    def __init__(self, input_space:Space, output_space:Space):
        self.input_space = input_space
        self.output_space = output_space

    def __call__(self, input):
        if not self.input_space.contains(input):
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
