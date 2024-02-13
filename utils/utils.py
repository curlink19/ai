import os
import shutil
import torch
from functools import wraps
from torch import nn
from torch.utils.data import default_collate
import numpy as np
import gc


def clear_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)


def pack(x, collate=False):
    if collate:
        x = default_collate([x])

    if isinstance(x, (list, tuple)):
        return x
    else:
        return (x,)


def eval_no_grad(func):
    """
    Decorator for both doing model.eval() and torch.no_grad()

    !model is the first argument!
            func(model, ...)
    """

    @wraps(func)
    def wrapper(model: nn.Module, *args, **kwargs):
        with torch.no_grad():
            model.eval()
            return func(model, *args, **kwargs)

    return wrapper


def to_numpy(x):
    """
    First, it will try to convert x to list.

    If x[0] is List, converts x[i] to numpy array
    Else, if x is List, converts x to numpy array
    """

    try:
        x = list(x)
    finally:
        pass

    if isinstance(x[0], list):
        return [np.array(val) for val in x]

    if isinstance(x, list):
        return np.array(x)

    return x


def gc_after(func):
    """
    Invokes gc.collect() after func
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        gc.collect()

    return wrapper


def gc_before(func):
    """
    Invokes gc.collect() before func
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        func(*args, **kwargs)

    return wrapper


def all_occurrences_generator(text: str, substr: str):
    """
    No overlaps.
    """
    current = 0
    while current != -1:
        current = text.find(substr, current)
        if current != -1:
            yield current
            current += len(substr)
