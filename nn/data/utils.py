from typing import Iterable, Any, List, Union
import torch
from torch.utils.data import random_split, Dataset, DataLoader, IterableDataset
from nn.configs.train_configs import TrainConfig


def random_split_dataset(
    dataset: Union[IterableDataset, Dataset],
    config: TrainConfig,
    to_loaders: bool = False,
):
    if config.valid_share is None:
        if to_loaders:
            return (
                DataLoader(
                    dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
                ),
                None,
            )
        else:
            return dataset, None

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


def to_device(device: torch.device, tensors: Any) -> Any:
    """
    While for modules .to(device) is inplace, it is not correct for Tensors.
    """
    if isinstance(tensors, (list, tuple)):
        result = []
        for tensor in tensors:
            result.append(tensor.to(device))
        return result
    return tensors.to(device)


def is_dataset(obj: Any) -> bool:
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")


def is_iterable_dataset(obj: Any) -> bool:
    return hasattr(
        obj,
        "__iter__",
    )
