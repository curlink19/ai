from typing import List
from rl.environment.types import Transition, State


class Stopper:
    def __call__(self, transition: Transition):
        raise NotImplementedError

    def alive(self) -> bool:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class TerminalStateStopper(Stopper):
    def __init__(self, terminal_states: List[State]):
        self.terminal_states = terminal_states
        self.is_alive = True

    def __call__(self, transition: Transition):
        if transition.next_state in self.terminal_states:
            self.is_alive = False

    def alive(self) -> bool:
        return self.is_alive

    def reset(self):
        self.is_alive = True
