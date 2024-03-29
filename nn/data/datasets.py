import pandas as pd
from tqdm import tqdm

import datasets
import torch
from torch import nn
from typing import Any, List, Tuple, Iterable, Union, Optional
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset  # noqa name troubles, be accurate
import random

from nn.data.preprocessors import Preprocessor
from nn.data.filters import Filter
from nn.models.tokenizers import Tokenizer
from nn.models.utils import evaluate
from nn.data.utils import to_device, is_dataset, is_iterable_dataset
from nn.data._iterators import ApplyFilterIterator
from utils.utils import gc_after


class ApplyFilter(IterableDataset):
    def __init__(
        self, dataset: Dataset, filter: Filter, logs=False
    ):  # noqa normal name
        super(IterableDataset).__init__()

        self.dataset = dataset
        self.filter = filter

        self.pbar = None
        if logs:
            self.pbar = tqdm(total=len(dataset))  # noqa must be implemented

    def __iter__(self):
        return ApplyFilterIterator(
            dataset=self.dataset, filter=self.filter, pbar=self.pbar
        )

    def __del__(self):
        if self.pbar is not None:
            self.pbar.close()


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
    """
    Tokenizing may be slow.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Tokenizer,
        log_interval: Optional[int] = None,
    ):
        super(Dataset).__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.log_interval = log_interval

    def __len__(self):
        return len(self.dataset)  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Any:
        if self.log_interval is not None and idx % self.log_interval == 0:
            print(f"{idx}")

        return self.tokenizer(self.dataset[idx])


class ApplyTokenizerToTupleElement(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Tokenizer, index: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.index = index

    def __len__(self):
        return len(self.dataset)  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Any:
        x = list(self.dataset[idx])
        x[self.index] = self.tokenizer(x[self.index])
        return x


class ListDataset(Dataset):
    def __init__(self, array: List[Any]):
        super(Dataset).__init__()

        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx: int) -> Any:
        return self.array[idx]

    @gc_after
    def clip(self, length: int, shuffle=True) -> None:
        if shuffle:
            random.shuffle(self.array)

        self.array = self.array[:length]


class MergedDataset(Dataset):
    def __init__(self, array: List[Dataset]):
        super(Dataset).__init__()

        self.array = array

    def __len__(self):
        return min(len(x) for x in self.array)  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return tuple(x[idx] for x in self.array)

    def to_hf_dataset(self, label_names: list[str]) -> datasets.Dataset:
        return datasets.Dataset.from_pandas(
            pd.DataFrame(
                data=[
                    dict(
                        [
                            (label_names[j], self.array[j][i])
                            for j in range(len(label_names))
                        ]
                    )
                    for i in range(len(self))
                ]
            )
        )


class HuggingFaceDictDataset(Dataset):
    def __init__(
        self,
        name: str,
        target_columns: Union[list[str], str],
        token: Optional[str] = None,
        length: Optional[int] = None,
        from_disk: bool = False,
    ):
        super(Dataset).__init__()

        if not from_disk:
            if token is None:
                self.dataset = load_dataset(name, split="train")
            else:
                self.dataset = load_dataset(name, split="train", token=token)
        else:
            self.dataset = datasets.load_from_disk(name)

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


class SequentialDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets

    def __len__(self):
        return sum(
            len(dataset) for dataset in self.datasets  # noqa, must be implemented
        )  # noqa, must be implemented

    def __getitem__(self, idx: int) -> Any:
        for dataset in self.datasets:
            if idx < len(dataset):  # noqa, must be implemented
                return dataset[idx]
            else:
                idx -= len(dataset)  # noqa, must be implemented
        assert False, "idx is too great"


def merge_datasets(
    datasets: Iterable[Union[Dataset, IterableDataset]], shuffle=True
) -> ListDataset:
    array = []

    for dataset in datasets:
        array.extend(to_list_dataset(dataset).array)

    if shuffle:
        random.shuffle(array)

    return ListDataset(array)


def to_list_dataset(
    dataset: Union[IterableDataset, Dataset], logs=False
) -> ListDataset:
    array = []

    if is_dataset(dataset):
        for i in (
            tqdm(
                range(len(dataset)),  # noqa must be implemented
            )
            if logs
            else range(len(dataset))  # noqa must be implemented
        ):
            array.append(dataset[i])
    else:
        assert is_iterable_dataset(dataset)
        assert not logs, "not yet implemented"

        for x in dataset:
            array.append(x)

    return ListDataset(array)


def spawn_clones(
    dataset: Union[IterableDataset, Dataset], times: int, iterate: bool = False
) -> ListDataset:
    if iterate:
        result = ListDataset([])
        for i in range(times):
            result.array.extend(to_list_dataset(dataset, logs=True).array)
    else:
        result = to_list_dataset(dataset)
        result.array = result.array * times
    return result


def push_to_hf(
    dataset: datasets.Dataset,
    name: str,
    append=False,
    token: str = None,
):
    if append:
        dataset = datasets.concatenate_datasets(
            [dataset, load_dataset(name, split="train", token=token)]
        )
    dataset.push_to_hub(name, token=token)


def add_constant_label(dataset: Dataset, label: Any) -> MergedDataset:
    return MergedDataset(
        [dataset, ListDataset([label] * len(dataset))]  # noqa must be implemented
    )
