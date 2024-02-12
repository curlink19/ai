from torch.utils.data import Dataset
from nn.data.filters import Filter


class ApplyFilterIterator:
    def __init__(self, dataset: Dataset, filter: Filter):
        self.dataset = dataset
        self.filter = filter
        self.current = 0

    def next(self):
        for i in range(len(self.dataset) - current):  # noqa must be implemented
            value = self.dataset[self.current + i]
            if self.filter(value):
                self.current += i + 1
                return value
        raise StopIteration
