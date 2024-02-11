from typing import Any, List, Tuple, Iterable
from torch.utils.data import Dataset
from datasets import load_dataset
import random

from nn.data.preprocessors import Preprocessor
from nn.models.tokenizers import Tokenizer


class ApplyPreprocessors(Dataset):
    """
    Apply preprocessors in list order.
    """

    def __init__(self, dataset: Dataset, preprocessors: list[Preprocessor]):
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
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Any:
        return self.tokenizer(self.dataset[idx])


class ListDataset(Dataset):
    def __init__(self, array: List[Any]):
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
        self.array = array

    def __len__(self):
        return min(len(x) for x in self.array)  # noqa

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return tuple(x[idx] for x in self.array)


class HuggingFaceDictDataset(Dataset):
    def __init__(self, name: str, target_column: str, length: int = None):
        self.dataset = load_dataset(name, split="train")
        self.target_column = target_column

        if length is not None:
            self.length = length
        else:
            self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx][self.target_column]


def merge_datasets(datasets: Iterable[Dataset], shuffle=True) -> ListDataset:
    array = []
    for dataset in datasets:
        for i in range(len(dataset)):
            array.append(dataset[i])

    if shuffle:
        random.shuffle(array)

    return ListDataset(array)


def to_list_dataset(dataset: Dataset) -> ListDataset:
    array = []
    for i in range(len(dataset)):
        array.append(dataset[i])
    return ListDataset(array)
