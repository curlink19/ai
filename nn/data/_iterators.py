from typing import Optional

import tqdm
from torch.utils.data import Dataset
from nn.data.filters import Filter


class ApplyFilterIterator:
    def __init__(self, dataset: Dataset, filter: Filter, pbar: Optional[tqdm.std.tqdm]):
        self.dataset = dataset
        self.filter = filter
        self.pbar = pbar

        self.log_interval = len(dataset) / 100  # noqa in %
        self.current = 0

    def __next__(self):
        for i in range(len(self.dataset) - self.current):  # noqa must be implemented
            value = self.dataset[self.current + i]

            log_iters_upd = self.current + i - self.pbar.n
            if self.pbar is not None and log_iters_upd > self.log_interval:
                self.pbar.update(log_iters_upd)

            if self.filter(value):
                self.current += i + 1
                return value
        raise StopIteration
