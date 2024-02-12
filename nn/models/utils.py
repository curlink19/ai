from typing import Optional, Any
import torch
from torch import nn

from utils.utils import eval_no_grad, pack
from nn.data.utils import to_device


@eval_no_grad
def evaluate(
    model: nn.Module, x: Any, device: Optional[torch.device] = None
) -> torch.Tensor:
    x = to_device(model.device, pack(x, collate=True))
    return to_device(model.device if device is None else device, model(*x))
