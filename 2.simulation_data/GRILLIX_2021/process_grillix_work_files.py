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

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from pathlib import Path
from tcvx21.grillix_post.validation_writer_m import (
    convert_work_file_to_validation_netcdf,
)
from tcvx21 import read_from_json

work_directory = Path("work_files").absolute()
output_directory = Path(".").absolute()

# %%
convert_work_file_to_validation_netcdf(
    work_file=work_directory / "GRILLIX_favourable_1mm_standard_fuelling.nc",
    output_file=Path("GRILLIX_forward_field.nc"),
    simulation_hierarchy=read_from_json(Path("simulation_hierarchy.json")),
)

# %%
convert_work_file_to_validation_netcdf(
    work_file=work_directory / "GRILLIX_unfavourable_1mm_standard_fuelling.nc",
    output_file=Path("GRILLIX_reversed_field.nc"),
    simulation_hierarchy=read_from_json(Path("simulation_hierarchy.json")),
)

# %%
import tcvx21
import matplotlib.pyplot as plt

plt.style.use(tcvx21.style_sheet)
from tcvx21.record_c import Record

standard_dictionary = dict(
    forward_field=read_from_json(tcvx21.template_file),
    reversed_field=read_from_json(tcvx21.template_file),
)

tcv = {
    "forward_field": Record(
        tcvx21.experimental_reference_dir / "TCV_forward_field.nc", color="C0"
    ),
    "reversed_field": Record(
        tcvx21.experimental_reference_dir / "TCV_reversed_field.nc", color="C0"
    ),
}
grx = {
    "forward_field": Record("GRILLIX_forward_field.nc", color="C1"),
    "reversed_field": Record("GRILLIX_reversed_field.nc", color="C1"),
}

for field_direction, fd_dict in standard_dictionary.items():

    for region, r_dict in fd_dict.items():

        for measurement, m_dict in r_dict["observables"].items():

            try:
                standard_dictionary[field_direction][region]["observables"][measurement]
            except KeyError:
                pass

            if region == "RDPA":
                fig, axs = plt.subplots(ncols=2)
                tcv[field_direction].get_observable(region, measurement).plot(ax=axs[0])
                grx[field_direction].get_observable(region, measurement).plot(ax=axs[1])
                plt.suptitle(f"{region}:{measurement}")
            else:
                fig, ax = plt.subplots()
                tcv[field_direction].get_observable(region, measurement).plot(ax=ax)
                grx[field_direction].get_observable(region, measurement).plot(ax=ax)
                plt.title(f"{region}:{measurement}")

# %%
