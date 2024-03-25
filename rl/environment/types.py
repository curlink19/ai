from typing import NewType, Any, Tuple, List


State = NewType("State", Any)
Action = NewType("Action", Any)
Reward = NewType("Reward", Any)
Transition = NewType("Transition", Tuple[State, Action, Reward, State])
Trajectory = NewType("Trajectory", List[Transition])
