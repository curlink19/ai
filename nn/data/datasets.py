import torch
from torch import nn
from typing import Any, List, Tuple, Iterable, Union
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset  # noqa name troubles, be accurate
import random

from nn.data.preprocessors import Preprocessor
from nn.data.filters import Filter
from nn.models.tokenizers import Tokenizer
from nn.models.utils import evaluate
from nn.data.utils import to_device
from nn.data._iterators import ApplyFilterIterator


class ApplyFilter(IterableDataset):
    def __init__(self, dataset: Dataset, filter: Filter):
        super(IterableDataset).__init__()

        self.dataset = dataset
        self.filter = filter

    def __iter__(self):
        return ApplyFilterIterator(dataset=self.dataset, filter=self.filter)


class ApplyLabeling(Dataset):
    def __init__(self, dataset: Dataset, model: nn.Module, online=False):
        super(Dataset).__init__()

        self.dataset = dataset
        self.model = model
        self.online = online

        self.labels = None
        if not self.online:
            self.labels = [
                evaluate(self.model, self.dataset[i], torch.device("cpu"))
                for i in range(len(self.dataset))  # noqa must be implemented
            ]

    def __len__(self):
        return len(self.dataset)  # noqa must be implemented

    def __getitem__(self, idx: int) -> Any:
        if self.online:
            return evaluate(self.model, self.dataset[idx])
        else:
            return to_device(self.model.device, self.labels[idx])


class ApplyPreprocessors(Dataset):
    """
    Apply preprocessors in list order.
    """

    def __init__(self, dataset: Dataset, preprocessors: list[Preprocessor]):
        super(Dataset).__init__()

        self.dataset = dataset
        self.preprocessors = preprocessors

    def __len__(self):
        return len(self.dataset)  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Any:
        x: Any = self.dataset[idx]

        for preprocessor in self.preprocessors:
            x = preprocessor(x)

        return x


class ApplyTokenizer(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Tokenizer):
        super(Dataset).__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Any:
        return self.tokenizer(self.dataset[idx])


class ListDataset(Dataset):
    def __init__(self, array: List[Any]):
        super(Dataset).__init__()

        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx: int) -> Any:
        return self.array[idx]

    def clip(self, length: int, shuffle=True) -> None:
        if shuffle:
            random.shuffle(self.array)

        self.array = self.array[:length]


class MergedDataset(Dataset):
    def __init__(self, array: List[Dataset]):
        super(Dataset).__init__()

        self.array = array

    def __len__(self):
        return min(len(x) for x in self.array)  # noqa

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return tuple(x[idx] for x in self.array)


class HuggingFaceDictDataset(Dataset):
    def __init__(
        self, name: str, target_columns: Union[list[str], str], length: int = None
    ):
        super(Dataset).__init__()

        self.dataset = load_dataset(name, split="train")
        self.target_columns = target_columns

        if length is not None:
            self.length = length
        else:
            self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Any:
        if isinstance(self.target_columns, list):
            return tuple(self.dataset[idx][col] for col in self.target_columns)
        return self.dataset[idx][self.target_columns]


def merge_datasets(datasets: Iterable[Dataset], shuffle=True) -> ListDataset:
    array = []
    for dataset in datasets:
        for i in range(len(dataset)):  # noqa must be implemented
            array.append(dataset[i])

    if shuffle:
        random.shuffle(array)

    return ListDataset(array)


def to_list_dataset(dataset: Dataset) -> ListDataset:
    array = []
    for i in range(len(dataset)):  # noqa must be implemented
        array.append(dataset[i])
    return ListDataset(array)
