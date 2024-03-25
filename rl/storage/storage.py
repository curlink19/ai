from typing import List, Optional, Union
from rl.environment.types import Transition, Trajectory
from rl.inference.inference import Inference
from rl.inference.stoppers import Stopper


class Storage:
    """
    All information garnered from environment.
    """

    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        self.trajectories = [] if trajectories is None else trajectories

    def extend(self, trajectories: List[Trajectory]):
        self.trajectories.extend(trajectories)

    def append_from_inference(self, inference: Inference, stopper: Stopper):
        self.trajectories.append(inference(stopper))
