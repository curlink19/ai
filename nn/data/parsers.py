from typing import Union, Any
import json


def parse_tg_json(path: str) -> list[str]:
    result = []
    with open(path) as f:
        data = json.loads(f.read())
        for message in data["messages"]:
            text = message["text"]
            if isinstance(text, list):
                real_text = ""
                for x in text:
                    if isinstance(x, str):
                        real_text += x
                text = real_text
            result.append(text)
    return result


def parse_json_dict(path: str, target_features: Union[list[str], str]) -> list[Any]:
    result = []
    with open(path) as f:
        data = json.loads(f.read())
        for element in data:
            if isinstance(target_features, str):
                result.append(element[target_features])
            else:
                result.append(tuple(element[x] for x in target_features))
    return result
