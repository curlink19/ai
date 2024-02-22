import re
from nn.data.datasets import ListDataset, to_list_dataset
from torch.utils.data import Dataset


def get_corpora_with_separator(dataset: Dataset, separators: list[str]) -> ListDataset:
    dataset = to_list_dataset(dataset)
    array = []

    sep = "|".join(separators)

    for i in range(len(dataset)):
        array.extend(re.split(sep, dataset[i]))

    return ListDataset(array)
