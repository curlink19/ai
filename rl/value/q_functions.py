import numpy as np
from rl.environment.types import State, Action, Reward


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
