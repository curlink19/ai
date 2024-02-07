from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = None
    batch_size: int = None
    batches_accumulated: int = None
