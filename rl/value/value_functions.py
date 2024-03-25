from rl.environment.types import State, Reward


class ValueFunction:
    def __call__(self, state: State) -> Reward:
        raise NotImplementedError

    def move(self, state: State, value: Reward, lr: float):
        raise NotImplementedError
