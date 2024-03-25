from rl.environment.types import State, Action


class TransitionModel:
    """
    (T) from (S, A, T, R, P)
    Finite --> [0, __len__) for all
    """

    def __call__(self, state: State, action: Action) -> State:
        raise NotImplementedError
