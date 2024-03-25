from typing import List
import numpy as np
from rl.environment.types import State
from rl.value.q_functions import QFunction, FiniteQFunction
from rl.value.value_functions import ValueFunction


class Environment:
    """
    (S, A, P) from (S, A, T, R, P)
    Finite --> [0, __len__) for all

    Finite --> Sparse ->> Other
    """

    def get_q_function(self) -> QFunction:
        raise NotImplementedError

    def get_value_function(self) -> ValueFunction:
        raise NotImplementedError

    def get_initial_state(self) -> State:
        raise NotImplementedError


class FiniteEnvironment(Environment):
    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        initial_states: List[int],
        initial_state_probs: np.ndarray,
    ):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.initial_states = []
        self.initial_state_probs = initial_state_probs

    def get_initial_state(self):
        return np.random.choice(a=self.initial_states, p=self.initial_state_probs)
