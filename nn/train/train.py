from torch.utils.data import Dataset, DataLoader
from torch import nn

from nn.train.Trainer import Trainer
from nn.configs.TrainConfig import TrainConfig
from nn.data.utils import random_split_dataset, to_device
from utils.Logger import Logger
from utils.utils import clear_dir, pack, eval_no_grad


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, trainer: Trainer, logger: Logger
) -> float:
    """
    Returns average loss.
    """

    model.train()

    total_loss = 0.0
    for i, (X, y) in enumerate(dataloader):
        to_device(model.device, X, y)
        total_loss += logger.step(trainer.step(model(*pack(X)), y))

    trainer.end_epoch()

    return total_loss / len(dataloader)


@eval_no_grad
def compute_avg_loss(
    model: nn.Module, dataloader: DataLoader, trainer: Trainer
) -> float:
    """
    Returns average loss.
    """

    total_loss = 0.0
    for i, (X, y) in enumerate(dataloader):
        to_device(model.device, X, y)
        total_loss += trainer.get_loss(model(*pack(X)), y)

    return total_loss / len(dataloader)


def train(
    model: nn.Module, dataset: Dataset, config: TrainConfig, trainer: Trainer
) -> None:
    train_loader, valid_loader = random_split_dataset(dataset, config, to_loaders=True)
    model.to(config.device)

    clear_dir(config.log_dir)
    train_epoch_logger = Logger(
        log_dir=config.log_dir,
        group="Loss/epoch/train",
        text_logs=True,
        log_interval=1,
        step_bias=0,
        reduction="mean",
        title=("epoch", "loss"),
    )
    valid_epoch_logger = Logger(
        log_dir=config.log_dir,
        group="Loss/epoch/valid",
        text_logs=True,
        log_interval=1,
        step_bias=0,
        reduction="mean",
        title=("valid, epoch", "loss"),
    )
    train_step_logger = Logger(
        log_dir=config.log_dir,
        group="Loss/batch/train",
        text_logs=True,
        log_interval=config.log_interval,
        step_bias=0,
        reduction="mean",
        title=("batch", "loss"),
    )

    for epoch in range(1, config.epochs + 1):
        train_epoch_logger.step(
            train_one_epoch(model, train_loader, trainer, train_step_logger)
        )
        valid_epoch_logger.step(compute_avg_loss(model, valid_loader, trainer))
