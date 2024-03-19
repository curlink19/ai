import numpy as np
from functools import partial
from typing import Optional
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from tritonn.clients.client import AsyncInferenceClient


class AsyncGrpcInferenceClient(AsyncInferenceClient):
    def __init__(
        self,
        url: str = "localhost:8001",
        verbose: bool = False,
        timeout: Optional[float] = None,
    ):
        self.triton_client =
