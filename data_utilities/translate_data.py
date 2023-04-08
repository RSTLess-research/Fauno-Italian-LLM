#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from time import sleep
from typing import *

import numpy as np
from googletrans import Translator
from tqdm import tqdm


def clean_sentence(sentence: str) -> str:
    return (
        sentence.replace('"', "")
        .replace("[| Umano |]", "[|Umano|]")
        .replace("[| AI |]", "[|AI|]")
        .replace("[| Human |]", "[|Human|]")
    )


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r") as reader:
        return json.loads(reader.read())


def write_jsonl(path, obj: List[Dict]):
    with open(path, "w") as writer:
        json.dump(obj, writer, indent=4)


if __name__ == "__main__":
    dataset_name: str = "medical"
    translation_api: Translator = Translator()
    data = read_jsonl(f"data/{dataset_name}_chat_data.json")[0]
    output = []
    sleep_time = 0.3
    for conversation in tqdm(data, desc="Translating data"):
        sleep(sleep_time)
        topic = conversation["topic"]
        input = conversation["input"]
        while True:
            try:
                translated_topic = translation_api.translate(
                    topic, src="en", dest="IT"
                ).text
                break
            except Exception as e:
                print(e)
                sleep(10)
        while True:
            try:
                translated_input = translation_api.translate(
                    input, src="en", dest="IT"
                ).text
                break
            except Exception as e:
                print(e)
                sleep(10)
        output.append(
            {
                "topic": clean_sentence(translated_topic),
                "input": clean_sentence(translated_input),
            }
        )
        write_jsonl(f"data/{dataset_name}_chat_data_IT2.json", output)
