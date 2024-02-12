from typing import Any
import torch
from torch.nn.functional import softmax


class Filter:
    """
    Checks some condition.
    """

    @torch.no_grad()
    def __call__(self, value: Any) -> bool:
        raise NotImplementedError


class LikelihoodThresholdFilter(Filter):
    def __init__(self, threshold: float, logits=True):
        self.threshold = threshold
        self.logits = logits

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> bool:
        x = x.squeeze()

        assert len(x.shape) == 1, "batch must contain single element"

        if self.logits:
            x = softmax(x)

        return torch.max(x).item() > self.threshold


class AndFilter(Filter):
    def __init__(self, filters: list[Filter]):
        self.filters = filters

    @torch.no_grad()
    def __call__(self, x: Any) -> bool:
        for filter in self.filters:
            if not filter.__call__(x):
                return False
        return True
