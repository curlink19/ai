import numpy as np
from scipy.special import softmax
from rl.environment.types import State, Action
from rl.value.q_functions import FiniteActionsQFunction
from utils.utils import prob_flag


class Policy:
    def __call__(self, state: State) -> Action:
        raise NotImplementedError


class EpsGreedyPolicy(Policy):
    def __init__(self, q_function: FiniteActionsQFunction, eps: float = 0):
        self.q_function = q_function
        self.eps = eps

    def __call__(self, state: State) -> Action:
        if prob_flag(self.eps):
            return np.random.randint(len(self.q_function))

        array = self.q_function.for_all_actions(state)
        assert len(array.shape) == 1

        return np.argmax(array)


class BoltzmannPolicy(Policy):
    def __init__(self, q_function: FiniteActionsQFunction, tau: float = 1):
        self.q_function = q_function
        self.tau = tau

    def __call__(self, state: State) -> Action:
        array = self.q_function.for_all_actions(state)
        assert len(array.shape) == 1

        array = softmax(array / self.tau)

        return np.random.choice(a=len(self.q_function), p=array)
