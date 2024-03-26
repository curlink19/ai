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
            q_function.move(
                transition.state,
                transition.action,
                transition.reward
                + gamma
                * q_function(transition.next_state, self.policy(transition.next_state)),
                lr=lr,
            )


class OffPolicyTDControl:
    def __init__(self, value: ValueFunction):
        self.value = value

    def __call__(
        self, q_function: QFunction, trajectory: Trajectory, lr: float, gamma: float
    ):
        for transition in trajectory:
            q_function.move(
                transition.state,
                transition.action,
                transition.reward + gamma * self.value(transition.next_state),
                lr=lr,
            )
