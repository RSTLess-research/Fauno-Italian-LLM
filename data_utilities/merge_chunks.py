#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


# read all the json files in the folder_path and merge them into one json file
def merge_json_files(folder_path):
    json_files = [
        pos_json
        for pos_json in os.listdir(folder_path)
        if pos_json.endswith(".json")
    ]
    json_data = []
    for file in json_files:
        with open(os.path.join(folder_path, file)) as f:
            json_data.extend(json.load(f))
    return json_data


if __name__ == "__main__":
    folder_path = "data_utilities/data_stack/"
    merged_data = merge_json_files(folder_path)
    with open("data_2/stack_chat_data_IT2.json", "w") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(
        "Merging complete. The merged data is saved in 'data_ITA/stack_chat_data_IT2.json'"
    )
