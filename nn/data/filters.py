from typing import Any
import torch
from torch.nn import Module
from torch.nn.functional import softmax
import numpy as np

from nn.models.utils import evaluate


class Filter:
    """
    Checks some condition.
    """

    @torch.no_grad()
    def __call__(self, value: Any) -> bool:
        raise NotImplementedError


class LikelihoodThresholdFilter(Filter):
    """
    if cls == -1, then checking maxprob class
    """

    def __init__(self, model: Module, threshold: float, cls: int = -1, logits=True):
        self.model = model
        self.threshold = threshold
        self.cls = cls
        self.logits = logits

    @torch.no_grad()
    def __call__(self, x: Any) -> bool:
        x = evaluate(self.model, x)

        x = x.squeeze()
        assert len(x.shape) == 1, "batch must contain single element"

        if self.logits:
            x = softmax(x, dim=-1)

        if self.cls == -1:
            return torch.max(x).item() > self.threshold
        return x[self.cls].item() > self.threshold


class AndFilter(Filter):
    def __init__(self, filters: list[Filter]):
        self.filters = filters

    @torch.no_grad()
    def __call__(self, x: Any) -> bool:
        for filter in self.filters:
            if not filter.__call__(x):
                return False
        return True


class SkipShortStringsWithProbability(Filter):
    """
    prob: probability of skipping
    """

    def __init__(self, min_length: int = 1, prob: float = 1):
        self.min_length = min_length
        self.prob = prob

    def __call__(self, x: str) -> bool:
        if len(x) < self.min_length:
            if np.random.binomial(1, self.prob) == 1:
                return False
            return True
        return True
