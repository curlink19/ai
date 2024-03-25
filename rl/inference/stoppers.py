from rl.environment.types import Transition


class Stopper:
    def __call__(self, transition: Transition):
        raise NotImplementedError

    def alive(self) -> bool:
        raise NotImplementedError
