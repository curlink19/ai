import os
import shutil
import torch
from functools import wraps
from torch import nn


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
