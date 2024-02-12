from typing import List, Union
from gigachat import GigaChat

from nn.configs.api_configs import GigaConfig


class Giga:
    def __init__(self, config: GigaConfig):
        self.config = config
        self._tokens_used: int = 0

    def __call__(self, prompt: Union[str, dict]) -> Union[List[str], str]:
        if isinstance(prompt, dict):
            prompt["model"] = self.config.model

        with GigaChat(
            credentials=self.config.credentials,
            model=self.config.model,
            verify_ssl_certs=False,
        ) as giga:
            response = giga.chat(prompt)
            result = [
                response.choices[i].message.content
                for i in range(len(response.choices))
            ]
            self._tokens_used += sum(
                [
                    x.tokens
                    for x in giga.tokens_count(
                        input_=result + [str(prompt)], model=self.config.model
                    )
                ]
            )
            return result if self.config.all_choices else result[0]

    def tokens_used(self):
        return self._tokens_used
