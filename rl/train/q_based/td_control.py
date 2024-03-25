from rl.value.value_functions import ValueFunction
from rl.value.q_functions import QFunction
from rl.environment.types import Trajectory
from rl.policy.policies import Policy


class OnPolicyTDControl:
    def __init__(self, policy: Policy):
        self.policy = policy

    def __call__(
        self, q_function: QFunction, trajectory: Trajectory, lr: float, gamma: float
    ):
        for transition in trajectory:
            state, action, reward, next_state = transition
            q_function.move(
                state,
                action,
                reward + gamma * q_function(next_state, self.policy(next_state)),
                lr=lr,
            )


class OffPolicyTDControl:
    def __init__(self, value: ValueFunction):
        self.value = value

    def __call__(
        self, q_function: QFunction, trajectory: Trajectory, lr: float, gamma: float
    ):
        for transition in trajectory:
            state, action, reward, next_state = transition
            q_function.move(
                state,
                action,
                reward + gamma * self.value(next_state),
                lr=lr,
            )
