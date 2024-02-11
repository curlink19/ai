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
