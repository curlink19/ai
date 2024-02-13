from typing import Any
import re
import numpy as np

from utils.utils import all_occurrences_generator


class Preprocessor:
    """
    It is implied that he accepts single value (same as for tokenizer).
    """

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError


class RemoveEmojis(Preprocessor):
    def __init__(self):
        self.emojis = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+",
            re.UNICODE,
        )

    def __call__(self, x: str) -> str:
        return re.sub(self.emojis, "", x)


class RemoveHTML(Preprocessor):
    def __init__(self):
        self.pattern = re.compile("<.*?>")

    def __call__(self, x: str) -> str:
        return re.sub(self.pattern, "", x)


class RemoveSymbols(Preprocessor):
    def __init__(self, symbols: list[str]):
        self.symbols = symbols

    def __call__(self, x: str) -> str:
        for symbol in self.symbols:
            x = x.replace(symbol, "")
        return x


class AddPrefixWithProbability(Preprocessor):
    def __init__(self, prefixes: list[str], prob: float):
        self.prefixes = prefixes
        self.prob = prob

    def __call__(self, x: str) -> str:
        if np.random.binomial(1, self.prob) == 0:
            return x
        return np.random.choice(self.prefixes) + x


class ReplaceWithProbability(Preprocessor):
    def __init__(self, replacements: list[tuple[str, str]], prob: float):
        self.replacements = replacements
        self.prob = prob

    def __call__(self, x: str) -> str:
        for replacement in self.replacements:
            if np.random.binomial(1, self.prob) == 1:
                x = x.replace(replacement[0], replacement[1])
        return x
