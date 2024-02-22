from typing import Any, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer

from utils.utils import to_numpy


class Tokenizer:
    """
    It is implied that he accepts single value.
    """

    def __call__(self, x: Any) -> Any:
        """
        Perform tokenization that is used by the model.
        """
        raise NotImplementedError

    def encode(self, x: Any) -> Any:
        """
        Perform tokenization and returns list of vocab indices.
        """
        raise NotImplementedError

    def tokenize(self, x: Any) -> Any:
        """
        Only tokenize.
        """
        return self.encode(x)

    def decode(self, x: Any) -> Any:
        """
        Decode from encode result.
        """
        raise NotImplementedError

    def set_max_length(self, max_length: int) -> None:
        """
        Sets max_length of tokenizer.
        """
        raise NotImplementedError

    def get_max_length(self) -> int:
        """
        Returns max_length of tokenizer.
        """
        raise NotImplementedError


class HuggingFaceTokenizer(Tokenizer):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> tuple[Any, ...]:
        return to_numpy(
            self.tokenizer(x, padding="max_length", truncation=True).values()
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tokenize(self, x: str) -> list[str]:
        return self.tokenizer.tokenize(x)

    def decode(self, *args, **kwargs) -> str:
        return self.tokenizer.decode(*args, **kwargs)

    def set_max_length(self, max_length: int) -> None:
        self.tokenizer.model_max_length = max_length

    def get_max_length(self) -> int:
        return self.tokenizer.model_max_length

    def from_pretrained(self, name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name)
