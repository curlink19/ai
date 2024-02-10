import torch
from dataclasses import dataclass

from utils.globals import get_global_variable


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = None
    batch_size: int = None
    batches_accumulated: int = None
    valid_share: float = 0.2
    device: torch.device = torch.device("cuda:0")

    log_dir: str = get_global_variable("log_dir")
    text_logs: bool = False
    log_interval: int = 1

    min_epoch_for_storing: int = 1
