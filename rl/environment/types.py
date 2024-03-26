from collections import namedtuple
from typing import NewType, Any, Tuple, List


State = NewType("State", Any)
Action = NewType("Action", Any)
Reward = NewType("Reward", Any)
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))
Trajectory = NewType("Trajectory", List[Transition])
