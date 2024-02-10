import torch
from torch.utils.data import random_split, Dataset, DataLoader
from nn.configs.train_configs import TrainConfig


def random_split_dataset(
    dataset: Dataset, config: TrainConfig, to_loaders: bool = False
):
    train_data, valid_data = random_split(
        dataset, [1 - config.valid_share, config.valid_share]
    )

    if to_loaders:
        train_data = DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        valid_data = DataLoader(
            valid_data, batch_size=config.batch_size, shuffle=True, drop_last=True
        )

    return train_data, valid_data


def to_device(device: torch.device, *tensors) -> None:
    for x in tensors:
        if isinstance(x, (list, tuple)):
            to_device(device, *x)
        else:
            x.to(device)
