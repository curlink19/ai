import os
import shutil
import torch
from functools import wraps
from torch import nn
import numpy as np


def clear_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)


def pack(x):
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
