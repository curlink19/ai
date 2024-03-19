import sys
import asyncio
from typing import Optional
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from host.triton.clients.client import AsyncInferenceClient, AsyncResponse

class AsyncGrpcResponse(AsyncResponse):
    # waiting using asyncio.Condition
"""
1: https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment
2: https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_grpc_async_infer_client.py
3: https://docs.python.org/3/library/asyncio-sync.html#condition
4: https://superfastpython.com/asyncio-condition-variable/#Notify_Waiting_Coroutines
"""

class AsyncGrpcInferenceClient(AsyncInferenceClient):
    def __init__(
        self,
        url: str = "localhost:8001",
        verbose: bool = False,
        timeout: Optional[float] = None,
    ):
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose,
            )
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        self.timeout = timeout

    async def async_inference_request(self):
        self.ready = asyncio.Condition()
