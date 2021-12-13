"""
A class to write a standard NetCDF file from any arbitrary data source.
"""
from pathlib import Path
from netCDF4 import Dataset, Group
import numpy as np
from tcvx21 import Quantity


def strip_units(array_in):
    try:
        return array_in.magnitude
    except AttributeError:
        assert isinstance(array_in, np.ndarray)
        return array_in


class RecordWriter:
    def __init__(
        self,
        file_path: Path,
        descriptor: str,
        description: str,
        allow_overwrite: bool = False,
    ):
        """
        Initialises a new dataset object and prepares to write into it

        file_path should be a pathlib.Path object pointing to a valid filepath where to write the NetCDF file to.
        if allow_overwrite, then existing files will be overwritten, otherwise an error will be raised if the filepath
        already exists.

        descriptor should be a short descriptor. It is used to automatically produce a figure label

        description is not used for the analysis. It should be a string with a long-form description of the
        simulation contained in the validation file
        For simulations, it should detail things such as;
        * the code used to generate the data
        * what git hash was used to run the cases
        * the name and contact email address of who generated the data
        * how long the simulation was run before starting to gather statistics
        * how long statistics were gathered for
        * any "tricks" used to stabilise the simulations
        """

        assert isinstance(
            file_path, Path
        ), f"Should pass file_path as a Path object but received {type(file_path)}"

        assert isinstance(
            descriptor, str
        ), f"Should pass a string as descriptor, but received type {type(descriptor)}"

        assert isinstance(
            description, str
        ), f"Should pass description as a string but received {type(file_path)}"

        assert (
            file_path.suffix == ".nc"
        ), f"Should use a .nc suffix, but suffix was {file_path.suffix}"

        if file_path.exists():
            if not allow_overwrite:
                raise FileExistsError(
                    f"The requested file_path {file_path} already "
                    f"exists and allow_overwrite is False"
                )
            else:
                print(f"Overwriting {file_path}")
                file_path.unlink()

        self.file_path = file_path
        self.description = description
        self.descriptor = descriptor

    def write_data_dict(self, data_dict: dict, additional_attributes: dict = {}):
        """
        Recursively writes entries from a standard-formatted dictionary (based on the observables.json template) into
        a NetCDF file
        """
        dataset = Dataset(self.file_path, "w")

        dataset.description = self.description
        dataset.descriptor = self.descriptor
        for attribute, value in additional_attributes.items():
            setattr(dataset, attribute, value)

        for diagnostic in data_dict.keys():
            self.write_diagnostic(dataset, data_dict[diagnostic], diagnostic)

        dataset.close()

    def write_diagnostic(
        self, dataset: Dataset, diagnostic_dict: dict, diagnostic: str
    ):
        """
        Writes the contents of a diagnostic into the NetCDF
        """
        diagnostic_group = dataset.createGroup(diagnostic)

        for key, val in diagnostic_dict.items():
            if key in ["observables", "name"]:
                continue
            setattr(diagnostic_group, key, val)

        diagnostic_group.diagnostic_name = diagnostic_dict["name"]
        observables_group = diagnostic_group.createGroup("observables")

        for observable in diagnostic_dict["observables"].keys():
            try:
                self.write_observable(
                    observables_group,
                    diagnostic_dict["observables"][observable],
                    observable,
                )
            except Exception as e:
                print(
                    f"Failed to write {diagnostic_dict['name']}:{observable}. Reraising error"
                )
                raise e from None

    @staticmethod
    def plain_text_units(units: str):
        """
        Converts a unit string to long-form ASCII text
        """
        return f"{Quantity(1, units).units}"

    def write_observable(self, observables_group: Group, data: dict, observable: str):
        """
        Write a observable into the NetCDF
        """

        observable_group = observables_group.createGroup(observable)
        observable_group.observable_name = data["name"]

        dimensionality = data["dimensionality"]
        n_points = np.size(data["values"])
        has_error = np.size(data["errors"]) != 0

        observable_group.dimensionality = dimensionality
        observable_group.experimental_hierarchy = data["experimental_hierarchy"]

        if data["simulation_hierarchy"] > 0:
            observable_group.simulation_hierarchy = data["simulation_hierarchy"]

        if n_points == 0:
            print(f"Missing data for {observable_group.path}")
            return

        plain_text_units = self.plain_text_units(data["units"])

        if dimensionality == 0:
            observable_group.createDimension(dimname="point", size=1)
            value = observable_group.createVariable(
                varname="value", datatype=np.float64, dimensions=("point",)
            )
            value[:] = strip_units(data["values"])
            value.units = plain_text_units

            if has_error:
                error = observable_group.createVariable(
                    varname="error", datatype=np.float64, dimensions=("point",)
                )
                error[:] = strip_units(data["errors"])
                error.units = plain_text_units

        elif dimensionality == 1:
            observable_group.createDimension(dimname="points", size=n_points)
            value = observable_group.createVariable(
                varname="value", datatype=np.float64, dimensions=("points",)
            )

            assert np.size(data["Ru"]) == n_points

            value[:] = strip_units(data["values"])
            value.units = plain_text_units

            r_upstream = observable_group.createVariable(
                varname="Rsep_omp", datatype=np.float64, dimensions=("points",)
            )
            r_upstream[:] = data["Ru"]
            r_upstream.units = self.plain_text_units(data["Ru_units"])

            # Radial position -- not required, but nice for reference
            if "R" in data.keys():
                r = observable_group.createVariable(
                    varname="R", datatype=np.float64, dimensions=("points",)
                )
                r[:] = data["R"]
                r.units = self.plain_text_units(data["R_units"])

            if "Z" in data.keys():
                # Vertical position
                z = observable_group.createVariable(
                    varname="Z", datatype=np.float64, dimensions=("points",)
                )
                z[:] = data["Z"]
                z.units = self.plain_text_units(data["Z_units"])

            if has_error:
                error = observable_group.createVariable(
                    varname="error", datatype=np.float64, dimensions=("points",)
                )
                error[:] = strip_units(data["errors"])
                error.units = plain_text_units

        elif dimensionality == 2:
            # Write flattened RDPA data
            observable_group.createDimension(dimname="points", size=n_points)
            value = observable_group.createVariable(
                varname="value", datatype=np.float64, dimensions=("points",)
            )
            value[:] = strip_units(data["values"])
            value.units = plain_text_units

            assert (
                np.size(data["Ru"]) == n_points
            ), f"{n_points}, {np.size(data['Ru'])}, {data}"
            assert (
                np.size(data["Zx"]) == n_points
            ), f"{n_points}, {np.size(data['Zx'])}, {data}"

            # Upstream-mapped radial position (flux-surface label)
            r_upstream = observable_group.createVariable(
                varname="Rsep_omp", datatype=np.float64, dimensions=("points",)
            )

            r_upstream[:] = data["Ru"]
            r_upstream.units = self.plain_text_units(data["Ru_units"])

            # Radial position -- not required, but nice for reference
            if "R" in data.keys():
                r = observable_group.createVariable(
                    varname="R", datatype=np.float64, dimensions=("points",)
                )
                r[:] = data["R"]
                r.units = self.plain_text_units(data["R_units"])

            # Vertical position relative to the X-point
            zx = observable_group.createVariable(
                varname="Zx", datatype=np.float64, dimensions=("points",)
            )

            zx[:] = data["Zx"]
            zx.units = self.plain_text_units(data["Zx_units"])

            if "Z" in data.keys():
                # Vertical position
                z = observable_group.createVariable(
                    varname="Z", datatype=np.float64, dimensions=("points",)
                )
                z[:] = data["Z"]
                z.units = self.plain_text_units(data["Z_units"])

            if has_error:
                error = observable_group.createVariable(
                    varname="error", datatype=np.float64, dimensions=("points",)
                )
                error[:] = strip_units(data["errors"])
                error.units = plain_text_units

        else:
            raise NotImplementedError(
                f"Haven't implemented a write method for dimensionality {dimensionality} (yet)"
            )
