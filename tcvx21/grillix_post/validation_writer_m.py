"""
Data extraction from GRILLIX is a two-step process. The first step is to make a "work file" of the
observable signal at each time point and toroidal plane. The second is take statistics over this file, and
write a NetCDF which is compatible with the validation analysis

This module performs the second step
"""
from pathlib import Path
import xarray as xr
from netCDF4 import Dataset

import tcvx21
from tcvx21.record_c.record_writer_m import RecordWriter
from tcvx21.units_m import Quantity
from tcvx21.file_io.json_io_m import read_from_json

from tcvx21.analysis.statistics_m import (
    compute_statistical_moment_with_bootstrap,
    compute_statistical_moment,
    strip_moment,
)


def convert_work_file_to_validation_netcdf(
    work_file: Path,
    output_file: Path,
    simulation_hierarchy: dict,
    statistics_interval: int = 500,
):
    """
    Converts a work file from the WorkFileWriter into a StandardNetCDF which can be used
    with the validation analysis

    The default statistics_interval is 500 time points, which corresponds to about 1ms
    """
    assert work_file.exists() and work_file.suffix == ".nc"
    assert isinstance(
        simulation_hierarchy, dict
    ), f"simulation_hierarchy should be dict, but was {type(simulation_hierarchy)}"

    standard_dictionary = fill_standard_dict(
        work_file, statistics_interval, simulation_hierarchy
    )

    dataset = Dataset(work_file)

    time = dataset["time"][slice(-statistics_interval, None)]
    statistics_time = (
        Quantity(time, dataset["time"].units).max()
        - Quantity(time, dataset["time"].units).min()
    )

    additional_attributes = {}

    for attribute in [
        "toroidal_planes",
        "points_per_plane",
        "n_points_R",
        "n_points_Z",
        "timestep",
        "timestep_units",
        "particle_source",
        "particle_source_units",
        "power_source",
        "power_source_units",
    ]:

        additional_attributes[attribute] = getattr(dataset, attribute)

    additional_attributes["statistics_interval"] = statistics_time.magnitude
    additional_attributes["statistics_interval_units"] = str(
        f"{statistics_time.units:P}"
    )

    writer = RecordWriter(
        file_path=output_file,
        descriptor="GRX",
        description=dataset.description,
        allow_overwrite=True,
    )

    writer.write_data_dict(standard_dictionary, additional_attributes)


def Q(netcdf_array):
    """Converts netcdf arrays to Quantities"""
    return Quantity(netcdf_array.values, netcdf_array.units)


def fill_standard_dict(
    work_file: Path, statistics_interval: int, simulation_hierarchy: dict
) -> dict:
    """
    Iterates over the elements in the standard dictionary, and fills each observable
    """
    print("Filling standard dict")

    standard_dict = read_from_json(tcvx21.template_file)

    # Expand the dictionary template to also include a region around the X-point
    standard_dict["Xpt"] = {
        "name": "X-point",
        "observables": {
            "density": {"name": "Plasma density", "units": "1 / meter ** 3"},
            "electron_temp": {"name": "Electron temperature", "units": "electron_volt"},
            "potential": {"name": "Plasma potential", "units": "volt"},
        },
    }

    for observable in standard_dict["Xpt"]["observables"].values():
        observable["dimensionality"] = 2
        observable["experimental_hierarchy"] = -1
        observable["simulation_hierarchy"] = -1

    for diagnostic, diagnostic_dict in standard_dict.items():
        for observable, observable_dict in diagnostic_dict["observables"].items():
            write_observable(
                work_file,
                statistics_interval,
                simulation_hierarchy,
                diagnostic,
                observable,
                observable_dict,
            )

    print("Done")
    return standard_dict


def write_observable(
    work_file: Path,
    statistics_interval: int,
    simulation_hierarchy: dict,
    diagnostic_key: str,
    observable_key: str,
    output_dict: dict,
):
    """
    Calculates statistical moments and fills values into the standard_dictionary, for writing
    to a standard NetCDF
    """

    print(f"\tProcessing {diagnostic_key}:{observable_key}")

    observable_key, moment = strip_moment(observable_key)
    try:
        diagnostic = xr.open_dataset(work_file, group=diagnostic_key)
    except OSError as e:  # pragma: no cover
        # Catch an old name for the diagnostic
        if diagnostic_key == "FHRP":
            diagnostic_key = "RPTCV"
            diagnostic = xr.open_dataset(work_file, group=diagnostic_key)
        else:
            raise e from None
    try:
        observable = xr.open_dataset(work_file, group=f"{diagnostic_key}/observables")[
            observable_key
        ]
    except OSError:
        # Catch an old term for the observables
        observable = xr.open_dataset(work_file, group=f"{diagnostic_key}/measurements")[
            observable_key
        ]

    observable = observable.isel(tau=slice(-statistics_interval, None)).persist()

    output_dict["simulation_hierarchy"] = simulation_hierarchy[
        observable_key if moment != "lambda_q" else moment
    ]

    if diagnostic_key == "Xpt":
        # Don't need an error estimate for the X-point profile, and the memory requirements for bootstrapping this
        # can be very large
        value = compute_statistical_moment(observable, moment=moment)
        error = 0.0 * value
    else:
        value, error = compute_statistical_moment_with_bootstrap(
            observable, moment=moment
        )

    output_dict["values"] = Q(value).to(output_dict["units"]).magnitude
    output_dict["errors"] = Q(error).to(output_dict["units"]).magnitude

    for variable_key in diagnostic.variables.keys():
        variable = diagnostic[variable_key]
        output_key = variable_key.replace("Rsep", "Ru")

        output_dict[output_key] = variable.values
        output_dict[f"{output_key}_units"] = getattr(variable, "units", "")
