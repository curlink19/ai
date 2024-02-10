from typing import Any, List, Tuple
from torch.utils.data import Dataset

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


class MergedDataset(Dataset):
    def __init__(self, array: List[Dataset]):
        self.array = array

    def __len__(self):
        return min(len(x) for x in self.array)  # noqa

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        return tuple(x[idx] for x in self.array)
