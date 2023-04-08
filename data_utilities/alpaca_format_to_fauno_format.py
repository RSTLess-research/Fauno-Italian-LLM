#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script converts the data from the alpaca format to the one required by the Fauno model.

import json
from typing import *


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.loads(f.read())


def write_jsonl(path: str, obj: List[dict]) -> None:
    with open(path, "w") as writer:
        json.dump(obj, writer, indent=4)


if __name__ == "__main__":
    data: List[dict] = read_jsonl("data/alpaca_chat_data.json")

    output_clean: List[dict] = []
    for elem in data:
        topic = elem["instruction"].replace('"', "")
        options = elem["input"]
        input_conversation = f"The conversation between human and AI assistant.\n[|Human|] {topic}{' ' + options if options else ''}.\n[|AI|] {elem['output']}".replace(
            '"', ""
        )
        output_clean.append({"topic": topic, "input": input_conversation})

    write_jsonl("data_2/alpaca_chat_data_IT2.json", output_clean)
