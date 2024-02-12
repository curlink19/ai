import torch
from dataclasses import dataclass

from utils.globals import get_global_variable


@dataclass(frozen=True)
class GigaConfig:
    client_id: str = None
    client_secret: str = None
    credentials: str = None
    scope: str = None
    model: str = "GigaChat"

    all_choices: bool = False
