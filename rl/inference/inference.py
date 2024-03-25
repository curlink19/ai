from rl.environment.types import Action, Reward, State, Trajectory, Transition
from rl.environment.environment import Environment
from rl.environment.reward_models import RewardModel
from rl.environment.transition_models import TransitionModel
from rl.policy.policies import Policy
from rl.inference.stoppers import Stopper


class Inference:
    def __init__(
        self,
        environment: Environment,
        policy: Policy,
        reward_model: RewardModel,
        transition_model: TransitionModel,
    ):
        self.environment = environment
        self.policy = policy
        self.reward_model = reward_model
        self.transition_model = transition_model

    def __call__(self, stopper: Stopper) -> Trajectory:
        trajectory: Trajectory = []  # noqa typing
        state: State = self.environment.get_initial_state()

        while stopper.alive():
            action: Action = self.policy(state)
            next_state: State = self.transition_model(state, action)
            reward: Reward = self.reward_model(state, action, next_state)
            transition: Transition = (state, action, reward, next_state)  # noqa typing

            trajectory.append(transition)
            stopper(transition)

        return trajectory
