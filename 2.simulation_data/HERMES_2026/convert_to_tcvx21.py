#!/usr/bin/env python3
#
# Converts a pickle file created by the gather_data script into
# a NetCDF file consistent with other TCV-X21 datasets

from pathlib import Path
import tcvx21
from tcvx21.record_c.record_writer_m import RecordWriter

def write_x21_dataset(hermes_data: dict, output_file: Path):
    """
    Write data to NetCDF, in the format of the TCV-X21 datasets
    """

    result = {
        "LFS-LP": {
            "name": "Low-field-side target Langmuir probes",
            "hermes_location": "lfs",
            "observables": {
                "density": {
                    "name": "Plasma density",
                    "hermes_name": "Ne",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "electron_temp": {
                    "name": "Electron temperature",
                    "hermes_name": "Te",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "ion_temp": {
                    "name": "Ion temperature",
                    "hermes_name": "Ti",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "potential": {
                    "name": "Plasma potential",
                    "hermes_name": "phi",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "current": {
                    "name": "Parallel current",
                    "hermes_name": "Jpar",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "vfloat": {
                    "name": "Floating potential",
                    "hermes_name": "Vfl",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
                "jsat": {
                    "name": "Ion saturation current",
                    "hermes_name": "Jsat",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
            },
        },
        "HFS-LP": {
            "name": "High-field-side target Langmuir probes",
            "hermes_location": "hfs",
            "observables": {
                "density": {
                    "name": "Plasma density",
                    "hermes_name": "Ne",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "electron_temp": {
                    "name": "Electron temperature",
                    "hermes_name": "Te",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "ion_temp": {
                    "name": "Ion temperature",
                    "hermes_name": "Ti",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "potential": {
                    "name": "Plasma potential",
                    "hermes_name": "phi",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "current": {
                    "name": "Parallel current",
                    "hermes_name": "Jpar",
                    "hermes_factor": -1.0,  # Reverse sign
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "vfloat": {
                    "name": "Floating potential",
                    "hermes_name": "Vfl",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
                "jsat": {
                    "name": "Ion saturation current",
                    "hermes_name": "Jsat",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
            },
        },
        "FHRP": {
            "name": "Outboard midplane reciprocating probe",
            "hermes_location": "omp",
            "observables": {
                "density": {
                    "name": "Plasma density",
                    "hermes_name": "Ne",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "electron_temp": {
                    "name": "Electron temperature",
                    "hermes_name": "Te",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "ion_temp": {
                    "name": "Ion temperature",
                    "hermes_name": "Ti",
                    "experimental_hierarchy": -1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "potential": {
                    "name": "Plasma potential",
                    "hermes_name": "phi",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "vfloat": {
                    "name": "Floating potential",
                    "hermes_name": "Vfl",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
                "jsat": {
                    "name": "Ion saturation current",
                    "hermes_name": "Jsat",
                    "experimental_hierarchy": 1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
                "mach_number": {
                    "name": "Plasma velocity normalised to local sound speed",
                    "hermes_name": "Mpar",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 2,
                },
            },
        },
        "LFS-IR": {
            "name": "Low-field-side target infrared camera",
            "hermes_location": "lfs",
            "observables": {
                "q_parallel": {
                    "name": "Parallel heat flux",
                    "hermes_name": "qpar",
                    "units": "W/m^2",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                }
            }
        },
        "TS": {
            "name": "Thomson scattering at divertor entrance (Z<0)",
            "hermes_location": "TS",
            "observables": {
                "density": {
                    "name": "Plasma density",
                    "hermes_name": "Ne",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "electron_temp": {
                    "name": "Electron temperature",
                    "hermes_name": "Te",
                    "experimental_hierarchy": 2,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
                "ion_temp": {
                    "name": "Ion temperature",
                    "hermes_name": "Ti",
                    "experimental_hierarchy": -1,
                    "dimensionality": 1,
                    "simulation_hierarchy": 1,
                },
            },
        },
        "RDPA": {
            "name": "Reciprocating divertor probe array",
            "hermes_location": "RDPA",
            "observables": {
                "density": {
                    "name": "Plasma density",
                    "hermes_name": "Ne",
                    "experimental_hierarchy": 2,
                    "dimensionality": 2,
                    "simulation_hierarchy": 1,
                },
                "electron_temp": {
                    "name": "Electron temperature",
                    "hermes_name": "Te",
                    "experimental_hierarchy": 2,
                    "dimensionality": 2,
                    "simulation_hierarchy": 1,
                },
                "ion_temp": {
                    "name": "Ion temperature",
                    "hermes_name": "Ti",
                    "experimental_hierarchy": -1,
                    "dimensionality": 2,
                    "simulation_hierarchy": 1,
                },
                "potential": {
                    "name": "Plasma potential",
                    "hermes_name": "phi",
                    "experimental_hierarchy": 2,
                    "dimensionality": 2,
                    "simulation_hierarchy": 1,
                },
                "vfloat": {
                    "name": "Floating potential",
                    "hermes_name": "Vfl",
                    "experimental_hierarchy": 1,
                    "dimensionality": 2,
                    "simulation_hierarchy": 2,
                },
                "jsat": {
                    "name": "Ion saturation current",
                    "hermes_name": "Jsat",
                    "experimental_hierarchy": 1,
                    "dimensionality": 2,
                    "simulation_hierarchy": 2,
                },
                "mach_number": {
                    "name": "Plasma velocity normalised to local sound speed",
                    "hermes_name": "Mpar",
                    "experimental_hierarchy": 2,
                    "dimensionality": 2,
                    "simulation_hierarchy": 2,
                },
            },
        },
    }

    # Add Hermes-3 data
    for dname, diagnostic in list(result.items()):
        observables = diagnostic["observables"]
        location = diagnostic["hermes_location"]
        try:
            for oname, observable in observables.items():
                hermes_obs = hermes_data[observable["hermes_name"]][location]
                factor = observable.get("hermes_factor", 1.0)
                observable["units"] = hermes_obs["units"]
                observable["values"] = factor * hermes_obs["mean"]
                observable["errors"] = hermes_obs["std"]
                observable["Ru"] = hermes_obs["Ru"]
                observable["Ru_units"] = hermes_obs["Ru_units"]
                if observable['dimensionality'] == 2:
                    observable['Zx'] = hermes_obs['Zx']
                    observable['Zx_units'] = hermes_obs['Zx_units']
        except KeyError as e:
            print(f"Skipping diagnostic {dname} : {e}")
            del result[dname]

    def skew(obs):
        # Central moment from moment around origin
        mu3 = obs["mean3"] - 3*obs["mean"]*obs["mean2"] + 2*obs["mean"]**3
        return mu3 / obs["std"]**3

    def kurt(obs):
        mu4 = obs["mean4"] - 4*obs["mean"]*obs["mean3"] + 6*obs["mean"]**2 * obs["mean2"] - 3*obs["mean"]**4
        return mu4 / obs["std"]**4

    # Observables that correspond to moments of calculated quantities
    for location, diagnostic in [("lfs", "LFS-LP"),
                                 ("hfs", "HFS-LP"),
                                 ("omp", "FHRP"),
                                 ("RDPA", "RDPA")]:
        dimensionality = result[diagnostic]["observables"]["jsat"]["dimensionality"]
        # Ion saturation current

        obs = hermes_data["Jsat"][location]
        result[diagnostic]["observables"]["jsat_std"] = {
            "name": "Standard deviation of the ion saturation current",
            "units": obs["units"],
            "experimental_hierarchy": 1,
            "simulation_hierarchy": 2,
            "dimensionality": dimensionality,
            "Ru": obs["Ru"],
            "Ru_units": obs["Ru_units"],
            "values": obs["std"],
            "errors": 0 * obs["std"],
        }
        if "Zx" in obs:
            result[diagnostic]["observables"]["jsat_std"]["Zx"] = obs["Zx"]
            result[diagnostic]["observables"]["jsat_std"]["Zx_units"] = obs["Zx_units"]
            
        result[diagnostic]["observables"]["jsat_skew"] = {
            "name": "Skew of the ion saturation current",
            "units": "",
            "experimental_hierarchy": 1,
            "simulation_hierarchy": 2,
            "dimensionality": dimensionality,
            "Ru": obs["Ru"],
            "Ru_units": obs["Ru_units"],
            "values": skew(obs),
            "errors": 0 * skew(obs),
        }
        if "Zx" in obs:
            result[diagnostic]["observables"]["jsat_skew"]["Zx"] = obs["Zx"]
            result[diagnostic]["observables"]["jsat_skew"]["Zx_units"] = obs["Zx_units"]

        result[diagnostic]["observables"]["jsat_kurtosis"] = {
            "name": "Pearson kurtosis of the ion saturation current",
            "units": "",
            "experimental_hierarchy": 1,
            "simulation_hierarchy": 2,
            "dimensionality": dimensionality,
            "Ru": obs["Ru"],
            "Ru_units": obs["Ru_units"],
            "values": kurt(obs),
            "errors": 0 * kurt(obs),
        }
        
        if "Zx" in obs:
            result[diagnostic]["observables"]["jsat_kurtosis"]["Zx"] = obs["Zx"]
            result[diagnostic]["observables"]["jsat_kurtosis"]["Zx_units"] = obs["Zx_units"]

        # Floating potential

        obs = hermes_data["Vfl"][location]
        result[diagnostic]["observables"]["vfloat_std"] = {
            "name": "Standard deviation of the floating potential",
            "units": obs["units"],
            "experimental_hierarchy": 1,
            "simulation_hierarchy": 2,
            "dimensionality": dimensionality,
            "Ru": obs["Ru"],
            "Ru_units": obs["Ru_units"],
            "values": obs["std"],
            "errors": 0 * obs["std"],
        }
        if "Zx" in obs:
            result[diagnostic]["observables"]["vfloat_std"]["Zx"] = obs["Zx"]
            result[diagnostic]["observables"]["vfloat_std"]["Zx_units"] = obs["Zx_units"]

        # Current
        if location in hermes_data["Jpar"]:
            obs = hermes_data["Jpar"][location]
            result[diagnostic]["observables"]["current_std"] = {
                "name": "Standard deviation of the parallel current",
                "units": obs["units"],
                "experimental_hierarchy": 1,
                "simulation_hierarchy": 1,
                "dimensionality": dimensionality,
                "Ru": obs["Ru"],
                "Ru_units": obs["Ru_units"],
                "values": obs["std"],
                "errors": 0 * obs["std"],
            }
            if "Zx" in obs:
                result[diagnostic]["observables"]["current_std"]["Zx"] = obs["Zx"]
                result[diagnostic]["observables"]["current_std"]["Zx_units"] = obs["Zx_units"]


    additional_attributes = {}

    writer = RecordWriter(
        file_path=output_file,
        descriptor="Hermes-3",
        description="Hermes-3 simulation dataset",
        allow_overwrite=True,
    )
    writer.write_data_dict(result, additional_attributes)


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description="Convert Hermes-3 pickle file into TCV-X21 NetCDF file"
    )

    parser.add_argument(
        "pickle_file_path", type=str, help="Pickle file containing Hermes-3 data"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="hermes_tcvx21_data.nc",
        help="Output NetCDF file in TCV-X21 format",
    )

    args = parser.parse_args()

    with open(args.pickle_file_path, "rb") as f:
        data = pickle.load(f)

    write_x21_dataset(data, Path(args.output))
