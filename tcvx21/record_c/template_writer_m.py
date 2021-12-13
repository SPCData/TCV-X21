"""
We want to make sure that each data source -- the experiment (TCV) and the different simulations -- 
all use the same data structure. This helps to ensure that no data is missing from the data source

By using a data template, we can simplify the post-processing routines since all the data sources
will be structured the same way

We do this by first mapping a data-source onto a standard dictionary, and then writing that standard
dictionary into a NetCDF file.

The standard dictionary is defined via the observables.json file. This is a simple .json file, 
which is defined to match the dictionary in this file

The structure of the dictionary is as follows

Top-level keys give observable positions/diagnostics
    Each key of diagnostics gives a different diagnostic
    Each diagnostic has
        name
        R_sample_points and Z_sample_points which give sample points for the reference shot.
        N.b. These aren't necessarily the comparison points, they're just points where you can
        take the data from, and then the routines will interpolate to the observable positions
    observables from that diagnostic
    Each observable in observables has
        name
        dimensionality (i.e. 0D, 1D or 2D)
        units as a pint-compatible string
        experimental_hierarchy primacy hierarchy for the experiment
        simulation_hierarchy (set to an error-flag of -1) which should be set by each code
        values giving the magnitude of the values in units of units
        errors giving the magnitude of the uncertainty in units of units
    For 1D data,
        Ru giving the upstream-mapped radial distance to the separatrix
        Ru_units giving the units of the upstream-mapped radial distance to the separatrix
    For 1D and 2D data,
        Zx giving the vertical position relative to the X-point
        Zx_units giving the units of the vertical position relative to the X-point
"""
from pathlib import Path
from scipy.io import savemat
from tcvx21 import write_to_json


observables = {
    "LFS-LP": {
        "name": "Low-field-side target Langmuir probes",
        "observables": {
            "density": {
                "name": "Plasma density",
                "units": "1/m^3",
                "experimental_hierarchy": 2,
            },
            "electron_temp": {
                "name": "Electron temperature",
                "units": "eV",
                "experimental_hierarchy": 2,
            },
            "ion_temp": {
                "name": "Ion temperature",
                "units": "eV",
                "experimental_hierarchy": -1,
            },
            "potential": {
                "name": "Plasma potential",
                "units": "V",
                "experimental_hierarchy": 2,
            },
            "current": {
                "name": "Parallel current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "current_std": {
                "name": "Standard deviation of the parallel current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat": {
                "name": "Ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_std": {
                "name": "Standard deviation of the ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_skew": {
                "name": "Skew of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "jsat_kurtosis": {
                "name": "Pearson kurtosis of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "vfloat": {
                "name": "Floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
            "vfloat_std": {
                "name": "Standard deviation of the floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
        },
    },
    "LFS-IR": {
        "name": "Low-field-side target infrared camera",
        "observables": {
            "q_parallel": {
                "name": "Parallel heat flux",
                "units": "W/m^2",
                "experimental_hierarchy": 2,
            }
        },
    },
    "HFS-LP": {
        "name": "High-field-side target Langmuir probes",
        "observables": {
            "density": {
                "name": "Plasma density",
                "units": "1/m^3",
                "experimental_hierarchy": 2,
            },
            "electron_temp": {
                "name": "Electron temperature",
                "units": "eV",
                "experimental_hierarchy": 2,
            },
            "ion_temp": {
                "name": "Ion temperature",
                "units": "eV",
                "experimental_hierarchy": -1,
            },
            "potential": {
                "name": "Plasma potential",
                "units": "V",
                "experimental_hierarchy": 2,
            },
            "current": {
                "name": "Parallel current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "current_std": {
                "name": "Standard deviation of the parallel current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat": {
                "name": "Ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_std": {
                "name": "Standard deviation of the ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_skew": {
                "name": "Skew of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "jsat_kurtosis": {
                "name": "Pearson kurtosis of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "vfloat": {
                "name": "Floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
            "vfloat_std": {
                "name": "Standard deviation of the floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
        },
    },
    "FHRP": {
        "name": "Outboard midplane reciprocating probe",
        "observables": {
            "density": {
                "name": "Plasma density",
                "units": "1/m^3",
                "experimental_hierarchy": 2,
            },
            "electron_temp": {
                "name": "Electron temperature",
                "units": "eV",
                "experimental_hierarchy": 2,
            },
            "ion_temp": {
                "name": "Ion temperature",
                "units": "eV",
                "experimental_hierarchy": -1,
            },
            "potential": {
                "name": "Plasma potential",
                "units": "V",
                "experimental_hierarchy": 2,
            },
            "jsat": {
                "name": "Ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_std": {
                "name": "Standard deviation of the ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_skew": {
                "name": "Skew of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "jsat_kurtosis": {
                "name": "Pearson kurtosis of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "vfloat": {
                "name": "Floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
            "vfloat_std": {
                "name": "Standard deviation of the floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
            "mach_number": {
                "name": "Plasma velocity normalised to local sound speed",
                "units": "",
                "experimental_hierarchy": 2,
            },
        },
    },
    "TS": {
        "name": "Thomson scattering at divertor entrance (Z<0)",
        "observables": {
            "density": {
                "name": "Plasma density",
                "units": "1/m^3",
                "experimental_hierarchy": 2,
            },
            "electron_temp": {
                "name": "Electron temperature",
                "units": "eV",
                "experimental_hierarchy": 2,
            },
            "ion_temp": {
                "name": "Ion temperature",
                "units": "eV",
                "experimental_hierarchy": -1,
            },
        },
    },
    "RDPA": {
        "name": "Reciprocating divertor probe array",
        "observables": {
            "density": {
                "name": "Plasma density",
                "units": "1/m^3",
                "experimental_hierarchy": 2,
            },
            "electron_temp": {
                "name": "Electron temperature",
                "units": "eV",
                "experimental_hierarchy": 2,
            },
            "ion_temp": {
                "name": "Ion temperature",
                "units": "eV",
                "experimental_hierarchy": -1,
            },
            "potential": {
                "name": "Plasma potential",
                "units": "V",
                "experimental_hierarchy": 2,
            },
            "jsat": {
                "name": "Ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_std": {
                "name": "Standard deviation of the ion saturation current",
                "units": "A/m^2",
                "experimental_hierarchy": 1,
            },
            "jsat_skew": {
                "name": "Skew of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "jsat_kurtosis": {
                "name": "Pearson kurtosis of the ion saturation current",
                "units": "",
                "experimental_hierarchy": 1,
            },
            "vfloat": {
                "name": "Floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
            "vfloat_std": {
                "name": "Standard deviation of the floating potential",
                "units": "V",
                "experimental_hierarchy": 1,
            },
            "mach_number": {
                "name": "Plasma velocity normalised to local sound speed",
                "units": "",
                "experimental_hierarchy": 2,
            },
        },
    },
}

diagnostics_1d = "LFS-LP", "HFS-LP", "LFS-IR", "FHRP", "TS"
diagnostics_2d = "RDPA"


def observables_template():
    """Fills in entries in the dictionary that are repeated and therefore left out in the definition"""
    observables_filled = observables.copy()

    for diagnostic_key, diagnostic in observables_filled.items():
        for observable in diagnostic["observables"].values():

            if diagnostic_key in diagnostics_1d:
                observable["dimensionality"] = 1

            elif diagnostic_key in diagnostics_2d:
                observable["dimensionality"] = 2

            else:
                raise NotImplementedError(
                    f"Diagnostics key {diagnostic_key} not recognised"
                )

            observable["simulation_hierarchy"] = -1
            observable["values"] = []
            observable["errors"] = []

            observable["Ru"] = []
            observable["Ru_units"] = "cm"

            if observable["dimensionality"] == 2:
                observable["Zx"] = []
                observable["Zx_units"] = "m"

    return observables_filled


def write_template_files(output_directory=Path(__file__).parent):
    """Writes template files for data analysis"""

    observables_filled = observables_template()

    # Write to JSON
    write_to_json(
        observables_filled,
        Path(output_directory) / "observables.json",
        allow_overwrite=True,
    )

    # Also write as a MATLAB struct
    # Since '-' is an illegal key in a MATLAB struct, replace it with '_'
    observables_for_mat = observables_filled.copy()
    for diagnostic_key in observables_filled.keys():

        if "-" in diagnostic_key:
            observables_for_mat[
                diagnostic_key.replace("-", "_")
            ] = observables_for_mat.pop(diagnostic_key)
            diagnostic_key = diagnostic_key.replace("-", "_")

        observables_group = observables_for_mat[diagnostic_key]["observables"]
        for observable_key in observables_group.keys():

            if "-" in observable_key:
                observables_group[
                    observable_key.replace("-", "_")
                ] = observables_group.pop(observable_key)

    savemat(
        Path(output_directory) / "observables.mat",
        {"CODE_NAME_field_direction": observables_for_mat},
    )


if __name__ == "__main__":
    write_template_files()
