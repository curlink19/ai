from rl.environment.types import State, Action, Reward


class RewardModel:
    """
    (R) from (S, A, T, R, P)
    Finite --> [0, __len__) for all
    """

    def __call__(self, state: State, action: Action, next_state: State) -> Reward:
        raise NotImplementedError
