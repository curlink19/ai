import asyncio
import numpy as np
from functools import partial
from typing import Optional, Union
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from host.triton.clients.clients import AsyncInferenceClient, AsyncResponse
from utils.utils import pack


class AsyncGrpcResponse(AsyncResponse):
    def __init__(self, output_names: list[str]):
        self.ready = asyncio.Condition()
        self.output_names = output_names
        self._data = []

    async def async_wait(self):
        async with self.ready:
            await self.ready.wait()

    def get(self):
        assert len(self._data) == 1, "some error occurred, maybe in asyncio logic"

        if isinstance(self._data[0], InferenceServerException):
            return None

        return [self._data[0].as_numpy(name) for name in self.output_names]


def async_grpc_callback(
    response: AsyncGrpcResponse, loop: asyncio.BaseEventLoop, result, error
):
    if error:
        response._data.append(error)  # noqa
    else:
        response._data.append(result)  # noqa

    async def coro():
        async with response.ready:
            return response.ready.notify_all()

    asyncio.run_coroutine_threadsafe(coro(), loop)


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
            self.triton_client = None

        self.timeout = timeout
        self.model_name = None
        self.input_names = None
        self.input_datatypes = None
        self.input_shapes = None
        self.output_names = None

    def set_input_config(
        self,
        model_name: str,
        name: Union[str, list[str]],
        shape: Union[list[int], list[list[int]]],
        datatype: Union[str, list[str]],
    ):
        if isinstance(name, str):
            assert isinstance(datatype, str) and isinstance(
                shape[0], int
            ), "wrong typing"
            name = pack(name)
            datatype = pack(datatype)
            shape = [shape]
        else:
            assert isinstance(datatype, list) and isinstance(
                shape[0], list
            ), "wrong typing"
            assert len(name) == len(shape) and len(name) == len(datatype), "wrong size"

        self.model_name = model_name
        self.input_names = name
        self.input_datatypes = datatype
        self.input_shapes = shape

    def set_output_config(self, name: Union[str, list[str]]):
        self.output_names = pack(name)

    async def async_inference_request(
        self, inputs: Union[np.ndarray, list[np.ndarray]]
    ) -> AsyncGrpcResponse:
        assert self.triton_client is not None, "Triton client failed to be created"
        assert self.model_name is not None, "Input config not set"
        assert self.output_names is not None, "Output config not set"

        inputs = pack(inputs)
        assert len(inputs) == len(self.input_shapes), "wrong number of inputs"

        client_inputs = []
        for name, shape, datatype, data in zip(
            self.input_names, self.input_shapes, self.input_datatypes, inputs
        ):
            client_inputs.append(
                grpcclient.InferInput(name=name, shape=shape, datatype=datatype)
            )
            client_inputs[-1].set_data_from_numpy(data)

        client_outputs = []
        for name in self.output_names:
            client_outputs.append(grpcclient.InferRequestedOutput(name=name))

        response = AsyncGrpcResponse(output_names=self.output_names)

        self.triton_client.async_infer(
            model_name=self.model_name,
            inputs=client_inputs,
            callback=partial(async_grpc_callback, response, asyncio.get_event_loop()),
            outputs=client_outputs,
            client_timeout=self.timeout,
        )

        return response
