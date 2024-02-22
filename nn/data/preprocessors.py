from typing import Any, Union

import re
import pymorphy3
from numpy.random import choice, randint

from utils.utils import all_occurrences_generator, prob_flag


class Preprocessor:
    """
    It is implied that he accepts single value (same as for tokenizer).
    """

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError


class Lower(Preprocessor):
    def __call__(self, x: str) -> str:
        return x.lower()


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
    def __init__(
        self,
        prefixes: Union[list[str], list[list[str]]],
        prob: float,
        upper_prob: float = 0.5,
    ):
        if not isinstance(prefixes[0], list):
            prefixes = [prefixes]

        self.prefixes = prefixes
        self.prob = prob
        self.upper_prob = upper_prob

    def __call__(self, x: str) -> str:
        if not prob_flag(self.prob):
            return x

        for current in reversed(self.prefixes):
            x = choice(current) + x

        if prob_flag(self.upper_prob):
            x = x[0].upper() + x[1:]

        return x


class ReplaceWithSynonymWithProbability(Preprocessor):
    def __init__(self, synonyms: list[str], prob: float):
        self.prob = prob
        self.morph = pymorphy3.MorphAnalyzer()

        self.cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]
        self.nums = ["sing", "plur"]

        self.synonyms = []
        for x in synonyms:
            word = self.morph.parse(x.lower())[0]
            current = []

            for case in self.cases:
                for num in self.nums:
                    current.append(" " + word.inflect({num, case}).word + " ")

            self.synonyms.append(current)

    def __call__(self, text: str) -> str:
        result = ""
        pos = 0

        occurs = []
        for i in range(len(self.synonyms)):
            for j in range(len(self.synonyms[i])):
                for k in all_occurrences_generator(text, self.synonyms[i][j]):
                    occurs.append((i, j, k))

        occurs = sorted(occurs, key=lambda x: x[-1])

        for occur in occurs:
            if occur[2] <= pos:
                continue

            result += text[pos : occur[2]]
            _len_text_synonym = len(self.synonyms[occur[0]][occur[1]])

            if prob_flag(self.prob):
                result += self.synonyms[randint(0, len(self.synonyms))][occur[1]]
            else:
                result += text[occur[2] : occur[2] + _len_text_synonym]

            pos = occur[2] + _len_text_synonym

        result += text[pos : pos + len(text)]

        return result


class RiddleWithProbability(Preprocessor):
    def __init__(
        self,
        separators: list[str],
        join_separators: list[str],
        min_length: int,
        delete_prob: float,
        replace_with_random_word_prob: float,
        insert_prob: float,
        random_words: list[str],
        random_insertions: list[str],
    ):
        self.separators = separators
        self.join_separators = join_separators
        self.sep = "|".join(self.separators)
        self.min_length = min_length
        self.delete_prob = delete_prob
        self.replace_with_random_word_prob = replace_with_random_word_prob
        self.insert_prob = insert_prob
        self.random_words = random_words
        self.random_insertions = random_insertions

    def __call__(self, x: str) -> str:
        array = []
        words = re.split(self.sep, x)

        if len(words) < self.min_length:
            return x

        for elem in words:
            if prob_flag(self.insert_prob):
                array.append(choice(self.random_insertions))

            if prob_flag(self.delete_prob):
                continue

            if prob_flag(self.replace_with_random_word_prob):
                array.append(choice(self.random_words))
                continue

            array.append(elem)

        if prob_flag(self.insert_prob):
            array.append(choice(self.random_insertions))

        result = ""
        for x in array:
            result += x + choice(self.join_separators)

        return result
