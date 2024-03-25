from typing import Any, Union, List
from torch.utils.data import Dataset, IterableDataset
from nn.data.datasets import Dataset, to_list_dataset


class AsyncResponse:
    async def async_wait(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        raise NotImplementedError


class AsyncInferenceClient:
    async def async_inference_request(self, *args, **kwargs) -> AsyncResponse:
        raise NotImplementedError


class ApplyDataset(AsyncInferenceClient):
    """
    The dataset input must be a list (as in ListDataset), which is given as an "array" argument.
    """

    def __init__(
        self,
        client: AsyncInferenceClient,
        array: List[Any],
        dataset: Union[Dataset, IterableDataset],
    ):
        self.client = client
        self.array = array
        self.dataset = dataset

    async def async_inference_request(self, x: Any):
        self.array.clear()
        self.array.append(x)

        response = await self.client.async_inference_request(self.dataset[0])

        return response
