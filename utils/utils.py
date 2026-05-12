import json
from pathlib import Path

import torch


def get_device() -> str:
    """
    Get the best available device for PyTorch operations.
    Returns:
        str: 'cuda' if a GPU is available, 'mps' for Apple Silicon, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_variables(list_names: list[str], file_path="dataset_variables.json") -> list[str]:
    """
    Read target variables from `file_path` file.
    Args:
        file_path: Path to the JSON file containing variable lists.
        list_names: List of keys to retrieve from the JSON.
    Returns:
        list: Combined list of target variable names as strings from the specified keys.
    """
    json_path = Path(__file__).parent / file_path
    with open(json_path, "r") as f:
        data = json.load(f)

    result = []
    for key in list_names:
        if key in data:
            result.extend(data[key])

    return result
