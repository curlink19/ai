from typing import List
import numpy as np
from rl.environment.types import State, Action, Reward
from rl.value.preprocessors import Preprocessor


class QFunction:
    def __call__(self, state: State, action: Action) -> Reward:
        raise NotImplementedError

    def move(self, state: State, action: Action, value: Reward, lr: float):
        raise NotImplementedError


class FiniteActionsQFunction(QFunction):
    def for_all_actions(self, state: State) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        """
        Number of actions, actions are [0, __len__)
        """
        raise NotImplementedError


class FiniteQFunction(FiniteActionsQFunction):
    def __init__(self, number_of_states: int, number_of_actions: int):
        self.array = np.zeros(
            shape=(number_of_states, number_of_actions), dtype=np.float32
        )

    def __call__(self, state: int, action: int):
        return self.array[state, action]

    def __len__(self):
        return self.array.shape[1]

    def move(self, state: int, action: int, value: float, lr: float = 1):
        self.array[state, action] += lr * (value - self.array[state, action])

    def for_all_actions(self, state: int) -> np.ndarray:
        return self.array[state, :]


class ApplyPreprocessors(QFunction):
    def __init__(self, preprocessors: List[Preprocessor], q_function: QFunction):
        self.preprocessors = preprocessors
        self.q_function = q_function

    def preprocess(self, state: State, action: Action):
        result = (state, action)
        for preprocessor in self.preprocessors:
            result = preprocessor(*result)
        return result

    def __call__(self, state: State, action: Action):
        state, action = self.preprocess(state, action)
        return self.q_function(state, action)

    def move(self, state: State, action: Action, value: Reward, lr: float):
        state, action = self.preprocess(state, action)
        self.q_function.move(state, action, value, lr)

    def __getattr__(self, item):  # TODO
        if item in dir(self.q_function):
            return object.__getattribute__(self.q_function, item)
        return object.__getattribute__(self, item)
