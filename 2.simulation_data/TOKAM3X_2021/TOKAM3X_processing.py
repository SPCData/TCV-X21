# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: tcv-x21
#     language: python
#     name: tcv-x21
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # TOKAM3X data processing
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
from tcvx21 import read_struct_from_file, summarise_tree_dict, read_from_json
from tcvx21.units_m import Quantity
import matplotlib.pyplot as plt

plt.style.use(tcvx21.style_sheet)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Read and structure the data from the source data file.

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
tk3x = read_struct_from_file(Path("simulation_data/TOKAM3X_forward_field.mat"))
summarise_tree_dict(tk3x, show_values=True)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Make measurement dictionaries from the template file for each field direction

# %% pycharm={"name": "#%%\n"}
standard_dictionary = read_from_json(tcvx21.template_file)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Copy values into the `measurements.json` standard_dictionary, applying a unit check
#
# Note that this unit check will only raise a warning, rather than error, if there is a
# unit conflict. This may have unintended consequences.

# %% pycharm={"name": "#%%\n"} tags=[]
field_direction_rename = {"forward_field": "Forw"}
region_rename = {
    "RDPA": "RDPA",
    "TS": "TS",
    "LFS-LP": "OSP",
    "LFS-IR": "OSP",
    "HFS-LP": "ISP",
    "FHRP": "OMID",
}
observable_rename = {
    "density": "N",
    "electron_temp": "Te",
    "ion_temp": "Ti",
    "potential": "PHI",
    "current": "Jpara",
    "current_std": None,
    "jsat": "Jsat",
    "jsat_std": None,
    "jsat_skew": None,
    "jsat_kurtosis": None,
    "vfloat": "Vf",
    "vfloat_std": None,
    "mach_number": "Mi",
    "q_parallel": "qtot",
    "lambda_q": None,
}

from tcvx21.units_m import Quantity

simulation_hierarchy = read_from_json(Path("simulation_data/simulation_hierachy.json"))


def strip_moment(observable_key: str):
    """Convert a observable name to a base observable and a statistical moment"""

    if observable_key.endswith("_std"):
        moment = "stdv"
        key = observable_key.rstrip("std").rstrip("_")
    elif observable_key.endswith("_skew"):
        moment = "skew"
        key = observable_key.rstrip("skew").rstrip("_")
    elif observable_key.endswith("_kurtosis"):
        moment = "kurt"
        key = observable_key.rstrip("kurtosis").rstrip("_")
    else:
        moment = "mean"
        key = observable_key

    return key, moment


def write_to_standard_dictionary(
    field_direction_key: str, region_key: str, observable_key: str
):
    """
    Check the value from the experiment and write into the standard dictionary

    A unit check is performed on the data. If the units do not match, a warning is
    raised and the units are used **from observables.json**
    This may have unexpected behaviour!

    Note that this uses the global values of `renamed_tcv` and `standard_dictionary`
    """

    try:
        output_ = standard_dictionary[region_key]["observables"][observable_key]
    except KeyError:
        return

    output_["simulation_hierarchy"] = simulation_hierarchy[observable_key]

    try:
        region_group = tk3x[field_direction_rename[field_direction_key]][
            region_rename[region_key]
        ]
    except KeyError:
        print(f"Missing {field_direction_key}:{region_key}:{observable_key}")
        return

    observable_key, moment = strip_moment(observable_key)

    if observable_rename[observable_key] is not None:
        observable = region_group[observable_rename[observable_key]][moment]

    else:
        print(f"Missing {field_direction_key}:{region_key}:{observable_key}")
        return

    if observable_key in ["mach_number", "current"] and moment != "mean":
        # It appears the helicity was accidently reversed for the statistical moments as well
        output_["values"] = -observable
    else:
        output_["values"] = observable

    if output_["dimensionality"] >= 1:
        output_["Ru"] = region_group["dRsep"]
        output_["Ru_units"] = "m"

    if output_["dimensionality"] >= 2:
        output_["R"] = region_group["R"]
        output_["R_units"] = "m"

        output_["Z"] = region_group["Z"]
        output_["Z_units"] = "m"

        x_point_z = -0.41411059
        output_["Zx"] = region_group["Z"] - x_point_z
        output_["Zx_units"] = "m"

    if output_["dimensionality"] > 2:
        raise NotImplementedError()


for region, r_dict in standard_dictionary.items():
    for observable, m_dict in r_dict["observables"].items():
        write_to_standard_dictionary("forward_field", region, observable)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Next, we write the standard dictionary to a NetCDF file, to allow easy reuse outside of
# Python.

# %% pycharm={"name": "#%%\n"}
from tcvx21.record_c import RecordWriter

writer = RecordWriter(
    file_path=Path(f"TOKAM3X_forward_field.nc"),
    descriptor="TOKAM3X",
    description=Path(
        f"simulation_data/TOKAM3X_forward_field_description.txt"
    ).read_text(),
    allow_overwrite=True,
)

writer.write_data_dict(standard_dictionary)

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### Compare TOKAM3X to experimental for sanity

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from tcvx21 import test_session

if not test_session:
    from tcvx21.record_c import Record

    tcv = Record(tcvx21.experimental_reference_dir / "TCV_forward_field.nc", color="C1")
    tkx = Record("TOKAM3X_forward_field.nc", color="C3")

    for region, r_dict in standard_dictionary.items():
        for measurement, m_dict in r_dict["observables"].items():

            try:
                standard_dictionary[region]["observables"][measurement]
            except KeyError:
                pass

            plt.figure()
            tcv.get_observable(region, measurement).plot()
            tkx.get_observable(region, measurement).plot()

            plt.title(f"{region}:{measurement}")
