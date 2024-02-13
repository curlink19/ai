from typing import Union, Optional
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch import nn

from nn.ensembles.dataset_ensembles import DatasetEnsemble
from nn.train.trainers import Trainer
from nn.configs.train_configs import TrainConfig
from nn.data.utils import (
    random_split_dataset,
    to_device,
    is_dataset,
    is_iterable_dataset,
)
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
        X = pack(X)
        X = to_device(model.device, X)
        y = to_device(model.device, y)

        total_loss += logger.step(trainer.step(model(*X), y))

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
        X = pack(X)
        X = to_device(model.device, X)
        y = to_device(model.device, y)

        total_loss += trainer.get_loss(model(*X), y)

    return total_loss / len(dataloader)


def train(
    model: nn.Module,
    dataset: Union[Dataset, IterableDataset],
    config: TrainConfig,
    trainer: Trainer,
    valid_ensemble: Optional[DatasetEnsemble] = None,
) -> None:
    if is_iterable_dataset(dataset):
        assert config.valid_share is None, "no valid for online-styled datasets"
    else:
        assert is_dataset(dataset)

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

    if valid_ensemble is not None:
        valid_ensemble_loggers = [
            Logger(
                log_dir=config.log_dir,
                group="Loss/epoch/valid_ensemble/" + valid_ensemble.names[i],
                text_logs=False,
                log_interval=1,
                step_bias=0,
                reduction="mean",
            )
            for i in range(len(valid_ensemble.names))
        ]
        valid_ensemble._generate_dataloaders(config)  # noqa here it is legal

    for epoch in range(1, config.epochs + 1):
        train_epoch_logger.step(
            train_one_epoch(model, train_loader, trainer, train_step_logger)
        )

        if config.valid_share is not None:
            valid_epoch_logger.step(compute_avg_loss(model, valid_loader, trainer))

        if valid_ensemble is not None:
            for i in range(len(valid_ensemble.datasets)):
                valid_ensemble_loggers[i].step(  # noqa check logic
                    compute_avg_loss(model, valid_ensemble.dataloaders[i], trainer)
                )
