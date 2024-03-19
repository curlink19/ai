class AsyncInferenceClient:
    def async_inference(self, *args, **kwargs):
        raise NotImplementedError

    async def async_wait(self, *args, **kwargs):
        raise NotImplementedError
