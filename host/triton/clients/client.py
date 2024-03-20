class AsyncResponse:
    async def async_wait(self, *args, **kwargs):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        raise NotImplementedError


class AsyncInferenceClient:
    async def async_inference_request(self, *args, **kwargs) -> AsyncResponse:
        raise NotImplementedError
