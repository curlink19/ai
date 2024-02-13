from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from nn.data.datasets import ListDataset, merge_datasets
from nn.configs.train_configs import TrainConfig


class DatasetEnsemble:
    def __init__(self, datasets: list[Dataset], names: Optional[list[str]] = None):
        self.datasets = datasets
        self.dataloaders = None

        if names is None:
            self.names = [str(i) for i in range(len(self.datasets))]
        else:
            self.names = names

    def _generate_dataloaders(self, config: TrainConfig):
        self.dataloaders = [
            DataLoader(
                dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
            )
            for dataset in self.datasets
        ]

    def extend(self, datasets: list[Dataset], names: Optional[list[str]] = None):
        if names is None:
            self.names += [str(i + len(self.datasets)) for i in range(len(datasets))]
        else:
            self.names += names

        self.datasets.extend(datasets)

    def plot_lengths(self, lines: Optional[list[float]] = None):
        plt.figure(figsize=(24, 8))

        if lines is not None:
            for line in lines:
                plt.axhline(y=line, color="r", linestyle="-")

        plt.bar(self.names, [len(x) for x in self.datasets])  # noqa must be implemented
        plt.title("size")
        plt.show()

    def clip(
        self, lengths: Union[int, list[int]], shuffle: bool = True, plot: bool = True
    ):
        if plot:
            self.plot_lengths(None if isinstance(lengths, list) else [lengths])

        for i in range(len(self.datasets)):
            current_length = lengths if isinstance(lengths, int) else lengths[i]
            self.datasets[i].clip(current_length, shuffle)  # noqa must be implemented

    def merge(self) -> ListDataset:
        return merge_datasets(self.datasets)
