from typing import List
import numpy as np
from rl.value.preprocessors import Preprocessor
from rl.environment.types import State, Reward


class ValueFunction:
    def __call__(self, state: State) -> Reward:
        raise NotImplementedError

    def move(self, state: State, value: Reward, lr: float):
        raise NotImplementedError


class FiniteValueFunction(ValueFunction):
    def __init__(self, number_of_states: int):
        self.array = np.zeros(shape=(number_of_states,), dtype=np.float32)

    def __call__(self, state: int) -> float:
        return self.array[state]  # noqa typing

    def move(self, state: State, value: Reward, lr: float = 1):
        self.array[state] += lr * (value - self.array[state])


class ApplyPreprocessors(ValueFunction):
    def __init__(
        self, preprocessors: List[Preprocessor], value_function: ValueFunction
    ):
        self.preprocessors = preprocessors
        self.value_function = value_function

    def preprocess(self, state: State):
        result = state
        for preprocessor in self.preprocessors:
            result = preprocessor(*result)
        return result

    def __call__(self, state: State):
        (state,) = self.preprocess((state,))
        return self.value_function(state)

    def move(self, state: State, value: Reward, lr: float):
        (state,) = self.preprocess((state,))
        self.value_function.move(state, value, lr)

    def __getattr__(self, item):  # TODO
        if item in dir(self.value_function):
            return object.__getattribute__(self.value_function, item)
        return object.__getattribute__(self, item)
