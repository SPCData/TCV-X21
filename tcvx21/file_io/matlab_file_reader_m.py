"""
Functions for reading MATLAB data '.mat' files

recomended usage is read_struct_from_file(filepath)
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path


def read_struct_from_file(filepath: Path, struct_name=None) -> dict:
    """
    Reads a struct called struct_name from a file at filepath
    """
    assert filepath.exists(), f"File not found {filepath.absolute()}"

    if struct_name is None:
        mat_file = loadmat(filepath, simplify_cells=True)
        struct_names = [
            key
            for key in mat_file
            if key not in ["__header__", "__version__", "__globals__"]
        ]

        if len(struct_names) > 1:
            raise RuntimeError(
                f"Multiple structs in file. Please pass struct_name as one of {struct_names}"
            )
        elif len(struct_names) == 0:
            raise RuntimeError(f"No structs in file")
        else:
            data = mat_file[struct_names[0]]

    else:
        data = loadmat(filepath, simplify_cells=True)[struct_name]

    return data
