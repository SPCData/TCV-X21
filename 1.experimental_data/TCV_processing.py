# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # TCV data processing
#
# Converts from a `.mat` MATLAB file which gives the TCV data, to a standardised
# Python dictionary defined in `tcvx21/record_c/observables.json`.
#
# This standard dictionary is then written to a standard NetCDF file.

# %% pycharm={"name": "#%%\n"}
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from pathlib import Path
import tcvx21
import numpy as np

# %%
expt = tcvx21.file_io.read_struct_from_file(
    Path("dataset_TCVX21_v2.mat"), struct_name="dataset"
)

expt = dict(forward_field=expt["Forw"], reversed_field=expt["Rev"])

# Since '-' is an illegal character in MATLAB struct-keys, we wrote these as '_'
# Reverse this change here
for field_direction, field_direction_dict in expt.items():
    diagnostic_keys = list(field_direction_dict.keys())

    for diagnostic in diagnostic_keys:
        field_direction_dict[diagnostic.replace("_", "-")] = field_direction_dict.pop(
            diagnostic
        )

# %%
for field_direction, field_direction_dict in expt.items():
    dss_dict = field_direction_dict["DSS"]

    for observable_data in dss_dict["observables"].values():
        observable_data["R"] = dss_dict["R"]
        observable_data["Z"] = dss_dict["Z"]
        observable_data["theta"] = dss_dict["theta"]

        (
            observable_data["R_units"],
            observable_data["Z_units"],
            observable_data["theta_units"],
        ) = ("m", "m", "rad")
        (
            observable_data["R_info"],
            observable_data["Z_info"],
            observable_data["theta_info"],
        ) = (
            "Radial location of the starting points of chords of view",
            "Vertical location of the starting points of chords of view",
            "Angle between chords of view and the negative R axis",
        )


# %% [markdown]
# The RDPA data is provided as arrays-of-arrays.
# Each element corresponds to a Z-position, and each sub-element corresponds to a $R^u - R^u_{sep}$ position
#
# We will expand this into a simple 1D array

# %%
def flatten_nested_arrays(input_array):
    """Flattens an array-of-arrays into a simple 1D array"""
    output_array = np.array([])
    for subarray in input_array:
        output_array = np.append(output_array, subarray)
    return output_array


def expand_nested_rdpa_arrays(observable_data):
    """Expands the Zx array to match the shape of the Rsep array"""

    rsep_coord = np.array([])
    zx_coord = np.array([])

    for z, rsep in zip(observable_data["Zx"], observable_data["Ru"]):

        rsep_coord = np.append(rsep_coord, rsep)
        zx_coord = np.append(zx_coord, z * np.ones_like(rsep))

    observable_data["Ru"] = rsep_coord
    observable_data["Zx"] = zx_coord
    observable_data["values"] = flatten_nested_arrays(observable_data["values"])
    observable_data["errors"] = flatten_nested_arrays(observable_data["errors"])

    shape = observable_data["values"].shape
    for key in ["Ru", "Zx", "values", "errors"]:
        assert observable_data[key].shape == shape


for field_direction, field_direction_dict in expt.items():
    for observable_data in field_direction_dict["RDPA"]["observables"].values():

        expand_nested_rdpa_arrays(observable_data)


# %%
# Load in the blank template
standard_dictionary = dict(
    forward_field=tcvx21.file_io.read_from_json(tcvx21.template_file),
    reversed_field=tcvx21.file_io.read_from_json(tcvx21.template_file),
)

# %% [markdown]
# We will add some additional information about the diagnostic positions into the standard dictionary.

# %%
for data_dict in standard_dictionary.values():

    data_dict["FHRP"]["Z"] = 0.0
    data_dict["FHRP"]["Z_units"] = "m"
    data_dict["FHRP"]["Z_info"] = "Vertical position of the FHRP measurement"

    data_dict["TS"]["R"] = 0.9
    data_dict["TS"]["R_units"] = "m"
    data_dict["TS"]["R_info"] = "Radial position of the TS measurement"

    data_dict["LFS-LP"]["Z"] = -0.75
    data_dict["LFS-LP"]["Z_units"] = "m"
    data_dict["LFS-LP"]["Z_info"] = "Vertical position of the LFS target"

    data_dict["LFS-IR"]["Z"] = -0.75
    data_dict["LFS-IR"]["Z_units"] = "m"
    data_dict["LFS-IR"]["Z_info"] = "Vertical position of the LFS target"

    data_dict["HFS-LP"]["R"] = 0.68
    data_dict["HFS-LP"]["R_units"] = "m"
    data_dict["HFS-LP"]["R_info"] = "Horizontal position of the HFS target"

# %% tags=[]
from tcvx21.units_m import Quantity, pint
import warnings


def write_to_standard_dictionary(
    field_direction: str, diagnostic: str, observable: str
):

    source = expt[field_direction][diagnostic]["observables"][observable]
    dest = standard_dictionary[field_direction][diagnostic]["observables"][observable]

    # Set all simulation hierarchies to -1 (no value)
    source["simulation_hierarchy"] = -1

    for key in source.keys():

        if key == "units" and len(source[key]) == 0:
            # Set unitless quantities to have an empty string as their 'units'
            source[key] = ""

        if key in ["values", "errors", "Ru", "Zx", "R", "Z", "theta"]:
            # Check that all of the static fields in the template are unchanged
            # Raise a warning if a mismatch is found (some of these occur due to strings changing in the round-trip to MATLAB)
            dest[key] = np.array(source[key])
        elif source[key] == dest[key]:
            pass
        else:
            # Take the value from the source, and warn
            warnings.warn(
                f"Source and destination did not match for static attribute {key}. Source was '{source[key]}' and dest was '{dest[key]}', for {field_direction}:{diagnostic}:{observable}"
            )

    shape = dest["values"].shape
    for key in ["values", "errors", "Ru", "Zx"]:
        if key in dest:
            val = dest[key]
            assert (
                val.shape == shape
            ), f"Array shape did not match for {field_direction}:{diagnostic}:{observable}:{key} -- {shape}, {val.shape}"
            assert np.issubdtype(
                val.dtype, np.number
            ), f"Array contained non-numerical values for {field_direction}:{diagnostic}:{observable}:{key}."


for field_direction, field_direction_dict in expt.items():
    for diagnostic, diagnostic_dict in field_direction_dict.items():

        for observable, observable_dict in diagnostic_dict["observables"].items():

            try:
                # Check if a KeyError is raised when trying to access the value
                standard_dictionary[field_direction][diagnostic]["observables"][
                    observable
                ]
            except KeyError:
                # If yes, skip this entry
                continue

            # Check that a corresponding key is found in the experimental data file
            expt[field_direction][diagnostic]["observables"][observable]

            write_to_standard_dictionary(field_direction, diagnostic, observable)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Next, we write the standard dictionary to a NetCDF file, to allow easy reuse outside of
# Python.

# %% pycharm={"name": "#%%\n"}
from tcvx21.record_c import RecordWriter

for field_direction in ["forward_field", "reversed_field"]:

    writer = RecordWriter(
        file_path=Path(f"TCV_{field_direction}.nc"),
        descriptor="TCV",
        description=Path(
            f"reference_scenario/TCV_{field_direction}_description.txt"
        ).read_text(),
        allow_overwrite=True,
    )

    writer.write_data_dict(standard_dictionary[field_direction])

# %%
