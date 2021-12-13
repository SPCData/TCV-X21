"""
Implementation of a reader for Fortran namelist readers, which can be used to interface with parameter files
"""

from collections import defaultdict
from os import name
from pathlib import Path
import f90nml
from tempfile import NamedTemporaryFile
import fnmatch


def read_fortran_namelist(path: Path):
    """
    Fortran namelist reader, using the f90nml module.

    If the '+' character is detected in the source file, a temporary file is made
    which removes '+'. This prevents a possible error in the namelist reading.
    """

    assert path.exists()

    with open(path, "r") as file:
        contents = file.read()

        if "+" in contents:
            contents = contents.replace("+", "")

            temp_path = Path(NamedTemporaryFile(delete=False).name)

            with open(temp_path, "w") as temp_file:
                temp_file.write(contents)

            namelist = f90nml_read_file(temp_path)

            # Remove the temporary file
            temp_path.unlink()

        else:
            namelist = f90nml_read_file(path)

    return convert_dict_to_defaultdict_recursive(namelist)


def f90nml_read_file(filename: Path):
    """
    Uses the f90nml library to read the namelist, and then returns a defaultdict of the result
    """

    namelist = f90nml.read(str(filename)).todict()

    return namelist


def convert_dict_to_defaultdict_recursive(input_dict):
    """
    Recursively converts all dictionaries, and values which are dictionaries, into defaultdicts
    """

    input_dict = defaultdict(list, input_dict)

    for key, item in input_dict.items():
        if isinstance(item, dict):
            input_dict[key] = convert_dict_to_defaultdict_recursive(item)

    return input_dict


def convert_params_filepaths(parameter_file_path: Path, params: dict):
    """
    Adds parameters in linked namelists to the base parameter dictionary
    """
    params_paths = {
        "grid_params_path": "params_grid",
        "map_params_path": "params_map",
        "init_params_path": "params_init",
        "trace_params_path": "params_trace",
        "tstep_params_path": "params_tstep",
        "physmod_params_path": "params_physmod",
        "bndconds_params_path": "params_bndconds",
        "bufferz_params_path": "params_bufferz",
        "floors_params_path": "params_floors",
        "multigrid_params_path": "params_multigrid",
        "pimsolver_params_path": "params_pimsolver",
        "penalisation_params_path": "params_penalisation",
        "diags_params_path": "params_diags",
        "srcsnk_params_path": "params_srcsnk",
        "tempdev_params_path": "params_tempdev",
        "neutrals_params_path": "params_neutrals",
        "iotrunk_params_path": "params_iotrunk",
    }

    for key, path in params["params_filepaths"].items():

        pointer_filepath = parameter_file_path.parent / path
        assert pointer_filepath.exists()
        pointer_params = read_fortran_namelist(pointer_filepath)

        if key == "equi_init_params_path":
            equi_params = fnmatch.filter(pointer_params.keys(), "equi_*_params")
            assert len(equi_params) == 1

            params[equi_params[0]] = pointer_params[equi_params[0]]
        else:
            new_key = params_paths[key]
            if not new_key in pointer_params.keys():
                print(f"No match for {new_key}. Setting to []")
            else:
                params[new_key] = pointer_params[new_key]

    return params
