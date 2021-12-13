"""
Routines for writing dictionaries to JSON-format files, and reading dictionaries from JSON-format files

Autonmatically handles numpy array i/o
"""

import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """
    From https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    Converts numpy arrays into lists
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def recursive_convert_list_to_array(data: dict) -> dict:
    """
    Converts all list elements in a nested dictionary to np.array
    """

    for key, value in data.items():

        if isinstance(value, list):
            data[key] = np.asarray(value)

        elif isinstance(value, dict):
            data[key] = recursive_convert_list_to_array(value)

    return data


def write_to_json(data: dict, filepath: Path, allow_overwrite: bool = False):
    """
    Writes a dictionary of data to a JSON file at filepath
    """

    assert (
        filepath.suffix == ".json"
    ), f"JSON files should be identified with the .json suffix"

    if not allow_overwrite:
        assert not filepath.exists()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def read_from_json(filepath: Path, convert_list_to_array: bool = True) -> dict:
    """
    Reads a dictionary of data from a JSON file at filepath, and if convert_list_to_array is True
    will convert all lists to np.array
    """

    assert filepath.exists(), f"File {filepath} not found"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if convert_list_to_array:
        data = recursive_convert_list_to_array(data)

    return data
