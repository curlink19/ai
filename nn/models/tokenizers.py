from typing import Any, Optional
from transformers.models.bert.tokenization_bert import BertTokenizer

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


class HuggingFaceBertTokenizer(Tokenizer):
    def __init__(self, tokenizer: Optional[BertTokenizer] = None):
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> tuple[Any, ...]:
        return to_numpy(self.tokenizer(x).values())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tokenize(self, x: str) -> list[str]:
        return self.tokenizer.tokenize(x)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def from_pretrained(self, name: str) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(name)
