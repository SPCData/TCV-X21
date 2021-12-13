"""
Data extraction from GRILLIX is a two-step process. The first step is to make a "work file" of the
observable signal at each tau point and toroidal plane. The second is take statistics over this file, and
write a NetCDF which is compatible with the validation analysis

This module performs the first step
"""
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from netCDF4 import Dataset
import numpy as np

import tcvx21
from tcvx21.units_m import Quantity, Dimensionless, convert_xarray_to_quantity
from tcvx21.file_io.json_io_m import read_from_json

from tcvx21.grillix_post.components import (
    Grid,
    Equi,
    Normalisation,
    read_snaps_from_file,
    get_snap_length_from_file,
    read_fortran_namelist,
    convert_params_filepaths,
    integrated_sources,
)

from tcvx21.grillix_post.lineouts import (
    OutboardMidplaneMap,
    outboard_midplane_chord,
    penalisation_contour,
    thomson_scattering,
    rdpa,
    xpoint,
)

from tcvx21.grillix_post.observables import (
    floating_potential,
    ion_saturation_current,
    sound_speed,
    compute_parallel_gradient,
    initialise_lineout_for_parallel_gradient,
    total_parallel_heat_flux,
    effective_parallel_exb_velocity,
)

xr.set_options(keep_attrs=True)
search_paths = [".", "trunk", ".."]


def filepath_resolver(directory: Path, search_file: str):
    """
    Checks relative paths specified in search_paths for a specified file or folder
    """

    for search_path in search_paths:
        search_directory = directory / search_path

        if search_directory.exists():
            if search_file in [file.name for file in search_directory.iterdir()]:
                found_file = search_directory / search_file

                assert found_file.exists()
                return found_file.absolute()

    raise FileNotFoundError(f"{search_file} not found in {directory}")


class WorkFileWriter:
    """
    Class to extract data from the raw GRILLIX snapshots.

    Writes to a "work file"
    """

    def __init__(
        self,
        file_path: Path,
        work_file: Path,
        equi_file: str = "TCV_ortho.nc",
        toroidal_field_direction: str = "reverse",
        data_length: int = 1000,
        n_points: int = 500,
        make_work_file: bool = True,
    ):
        """
        Initialises an object to store general information about a run, as well as lineouts for the tcvx21 case, and
        methods to calculate experimental quantities
        """
        assert toroidal_field_direction in [
            "forward",
            "reverse",
        ], f"toroidal_field_direction must be forward or reverse, but was {toroidal_field_direction}"

        self.file_path = Path(file_path)
        self.work_file = Path(work_file)
        self.data_length = data_length

        assert (
            self.file_path / "description.txt"
        ).exists(), f"Need to have a description.txt file in file_path"

        print("Setting up core analysis")
        self.grid = Grid(filepath_resolver(file_path, "vgrid.nc"))
        self.norm = Normalisation.initialise_from_normalisation_file(
            filepath_resolver(file_path, "physical_parameters.nml")
        )
        self.equi = Equi(
            equi_file=filepath_resolver(file_path, equi_file),
            penalisation_file=filepath_resolver(file_path, "pen_metainfo.nc"),
            flip_z=False if toroidal_field_direction == "reverse" else True,
        )

        parameter_filepath = filepath_resolver(file_path, "params.in")
        self.params = convert_params_filepaths(
            parameter_filepath, read_fortran_namelist(parameter_filepath)
        )

        self.diagnostic_to_lineout = {
            "LFS-LP": "lfs",
            "LFS-IR": "lfs",
            "HFS-LP": "hfs",
            "FHRP": "omp",
            "TS": "ts",
            "RDPA": "rdpa",
            "Xpt": "xpt",
        }

        print("Making lineouts")

        self.omp_map = OutboardMidplaneMap(self.grid, self.equi, self.norm)
        omp = outboard_midplane_chord(self.grid, self.equi, n_points=n_points)
        lfs = penalisation_contour(
            self.grid, self.equi, level=0.0, contour_index=0, n_points=n_points
        )
        hfs = penalisation_contour(
            self.grid, self.equi, level=0.0, contour_index=1, n_points=n_points
        )

        ts = thomson_scattering(
            self.grid, self.equi, tcvx21.thomson_coords_json, n_points=n_points
        )
        rdpa_ = rdpa(
            self.grid, self.equi, self.omp_map, self.norm, tcvx21.rdpa_coords_json
        )
        xpt = xpoint(self.grid, self.equi, self.norm)

        print("Tracing for parallel gradient")
        initialise_lineout_for_parallel_gradient(
            lfs,
            self.grid,
            self.equi,
            self.norm,
            npol=self.params["params_grid"]["npol"],
            stored_trace=file_path / f"lfs_trace_{n_points}.nc",
        )

        self.lineouts = {
            "omp": omp,
            "lfs": lfs,
            "hfs": hfs,
            "ts": ts,
            "rdpa": rdpa_,
            "xpt": xpt,
        }

        if not work_file.exists() and make_work_file:
            print("Initialising work file")
            self.initialise_work_file()
            print("Work file initialised")

        print("Done")

    def initialise_work_file(self):
        """
        Initialises a NetCDF file which can be iteratively filled with observables
        """

        dataset = Dataset(self.work_file, "w", format="NETCDF4")
        snaps_length = get_snap_length_from_file(file_path=self.file_path)

        dataset.first_snap = np.max((snaps_length - self.data_length, 0))
        print(
            f"Reading data from snap {dataset.first_snap} to last snap"
            f"(currently {snaps_length}), length {self.data_length}"
        )
        dataset.last_snap = dataset.first_snap

        self.write_summary(dataset.last_snap)

        dataset.createDimension(dimname="tau", size=None)
        dataset.createDimension(dimname="phi", size=self.params["params_grid"]["npol"])

        standard_dict = read_from_json(tcvx21.template_file)

        # Expand the dictionary template to also include a region around the X-point
        standard_dict["Xpt"] = {
            "observables": {
                "density": {"units": "1 / meter ** 3"},
                "electron_temp": {"units": "electron_volt"},
                "potential": {"units": "volt"},
            }
        }

        dataset.createVariable(varname="time", datatype=np.float64, dimensions=("tau",))
        dataset["time"].units = "milliseconds"
        dataset.createVariable(
            varname="snap_index", datatype=np.float64, dimensions=("tau",)
        )

        for diagnostic, diagnostic_dict in standard_dict.items():
            diagnostic_group = dataset.createGroup(diagnostic)

            lineout_key = self.diagnostic_to_lineout[diagnostic]
            diagnostic_group.lineout_key = lineout_key
            lineout = self.lineouts[diagnostic_group.lineout_key]

            diagnostic_group.createDimension(
                dimname="points", size=lineout.r_points.size
            )
            diagnostic_group.createVariable(
                varname="R", datatype=np.float64, dimensions=("points",)
            )
            diagnostic_group.createVariable(
                varname="Z", datatype=np.float64, dimensions=("points",)
            )
            diagnostic_group["R"][:] = lineout.r_points * self.norm.R0.to("m").magnitude
            diagnostic_group["R"].units = "meter"
            diagnostic_group["Z"][:] = (
                lineout.z_points
                * self.norm.R0.to("m").magnitude
                * (-1.0 if self.equi.flipped_z else 1.0)
            )
            diagnostic_group["Z"].units = "meter"

            diagnostic_group.createVariable(
                varname="rho", datatype=np.float64, dimensions=("points",)
            )
            diagnostic_group.createVariable(
                varname="Rsep", datatype=np.float64, dimensions=("points",)
            )
            diagnostic_group["rho"][:] = self.lineout_rho(lineout_key).values
            diagnostic_group["Rsep"][:] = (
                self.lineout_rsep(lineout_key).to("meter").magnitude
            )
            diagnostic_group["Rsep"].units = "meter"

            if hasattr(lineout, "coords"):
                diagnostic_group.createVariable(
                    varname="Zx", datatype=np.float64, dimensions=("points",)
                )
                diagnostic_group["Zx"][:] = lineout.coords["Zx"].to("m").magnitude
                diagnostic_group["Zx"].units = "meter"

            observables_group = diagnostic_group.createGroup("observables")

            for observable, observable_dict in diagnostic_dict["observables"].items():

                if (
                    observable.endswith("_std")
                    or observable.endswith("_skew")
                    or observable.endswith("_kurtosis")
                ):
                    # Don't calculate statistics or fits at this point
                    continue
                observable_var = observables_group.createVariable(
                    varname=observable,
                    datatype=np.float64,
                    dimensions=("tau", "phi", "points"),
                )

                observable_var.units = observable_dict["units"]

        dataset.close()

    def write_summary(self, snap):

        snaps = read_snaps_from_file(
            self.file_path, self.norm, time_slice=[snap], all_planes=True
        ).persist()
        sources = integrated_sources(
            self.grid, self.equi, self.norm, self.params, snaps
        )

        with Dataset(self.work_file, "a") as dataset:
            dataset.description = (self.file_path / "description.txt").read_text()
            dataset.toroidal_planes = snaps.sizes["phi"]
            dataset.points_per_plane = snaps.sizes["points"]
            dataset.n_points_R = self.grid.r_s.size
            dataset.n_points_Z = self.grid.z_s.size

            timestep = (
                self.params["params_tstep"]["dtau_max"] * self.norm.tau_0
            ).to_compact()
            dataset.timestep = timestep.magnitude
            dataset.timestep_units = str(f"{timestep.units:P}")

            dataset.particle_source = sources["density_source"].magnitude
            dataset.particle_source_units = str(f"{sources['density_source'].units:P}")
            dataset.power_source = sources["power_source"].magnitude
            dataset.power_source_units = str(f"{sources['power_source'].units:P}")

        snaps.close()

    def fill_work_file(self):
        """
        Iterates over the snaps dataset, to fill the complete dataset. Closes after each time segment, to prevent
        data loss. This isn't super efficient, but the data filling only has to be run once

        Summary elements like the description is updated each tau the script is executed
        """

        with Dataset(self.work_file, "a") as dataset:
            snaps_length = get_snap_length_from_file(file_path=self.file_path)
            first_snap = dataset.first_snap
            current_snap = dataset.last_snap
            last_snap = snaps_length

            print(
                f"First snap: {first_snap}, current_snap: {current_snap}, last_snap: {last_snap}"
            )

        for snap in range(current_snap, last_snap):
            t = snap - first_snap

            snaps = read_snaps_from_file(
                self.file_path, self.norm, time_slice=[snap], all_planes=True
            ).persist()

            with Dataset(self.work_file, "a") as dataset:
                # Open and close the dataset every write, to prevent data-loss
                # from timing out

                print(
                    f"Processing snap {snap} of {last_snap} (t={t}/{last_snap - first_snap})"
                )

                dataset["time"][t] = self.time(snaps=snaps).magnitude
                dataset["snap_index"][t] = dataset.first_snap + t

                for diagnostic_group in dataset.groups.values():
                    lineout_key = diagnostic_group.lineout_key
                    for observable_key, observable_var in diagnostic_group[
                        "observables"
                    ].variables.items():
                        observable = getattr(self, observable_key)
                        values = observable(lineout_key, snaps=snaps).compute()
                        values = values.transpose("tau", "phi", "interp_points")
                        values = (
                            convert_xarray_to_quantity(values)
                            .to(observable_var.units)
                            .magnitude
                        )

                        observable_var[t, :, :] = values

                dataset.last_snap = snap + 1

            snaps.close()

        print("Done")

    def time(self, snaps: xr.Dataset):
        """Returns an array of tau values (in ms)"""
        return convert_xarray_to_quantity(snaps.tau).to("ms")

    def lineout_rho(self, lineout: str) -> xr.DataArray:
        """Returns the flux-surface labels of a lineout"""
        lineout_ = self.lineouts[lineout]
        return self.equi.normalised_flux_surface_label(
            lineout_.r_points, lineout_.z_points, grid=False
        )

    def lineout_rsep(self, lineout: str) -> Quantity:
        """Returns the OMP-mapped distance along a lineout"""
        return convert_xarray_to_quantity(
            self.omp_map.convert_rho_to_distance(self.lineout_rho(lineout))
        )

    def density(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Returns the plasma density values along a lineout"""
        return self.lineouts[lineout].interpolate(snaps.density)

    def electron_temp(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Returns the electron temperature values along a lineout"""
        return self.lineouts[lineout].interpolate(snaps.electron_temp)

    def ion_temp(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Returns the ion temperature values along a lineout"""
        return self.lineouts[lineout].interpolate(snaps.ion_temp)

    def potential(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Returns the electrostatic potential values along a lineout"""
        return self.lineouts[lineout].interpolate(snaps.potential)

    def velocity(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Returns the ion velocity values along a lineout"""
        return self.lineouts[lineout].interpolate(snaps.velocity)

    def penalisation_direction_function(self, lineout: str) -> xr.DataArray:
        """Returns the penalisation direction function, which gives which field direction is 'towards' the target"""
        return self.lineouts[lineout].interpolate(self.equi.penalisation_direction)

    def current(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Returns the plasma current"""
        return self.lineouts[lineout].interpolate(
            snaps.current
        ) * self.penalisation_direction_function(lineout)

    def sound_speed(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Calculates and returns the sound speed"""
        return sound_speed(
            self.electron_temp(lineout, snaps), self.ion_temp(lineout, snaps), self.norm
        )

    def jsat(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """
        Calculates and returns the ion saturation current, with a factor of 0.5 depending on whether the
        lineout is immersed (for omp lineout, snaps) or a wall probe (for other hfs/lfs)
        """
        if lineout == "omp":
            return ion_saturation_current(
                self.density(lineout, snaps),
                self.sound_speed(lineout, snaps),
                self.norm,
                wall_probe=False,
            )
        elif lineout in ["hfs", "lfs", "rdpa"]:
            return ion_saturation_current(
                self.density(lineout, snaps),
                self.sound_speed(lineout, snaps),
                self.norm,
                wall_probe=True,
            )
        else:
            raise NotImplementedError(f"Lineout {lineout} not recognised")

    def vfloat(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Calculates and returns the floating potential"""
        return floating_potential(
            self.potential(lineout, snaps),
            self.electron_temp(lineout, snaps),
            self.norm,
        )

    def mach_number(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Calculates and returns the mach_number"""
        mach = self.velocity(lineout, snaps) / self.sound_speed(lineout, snaps)
        return mach.assign_attrs({"norm": Dimensionless})

    def electron_temp_parallel_gradient(self, lineout: str, snaps: xr.Dataset):
        """
        Parallel gradient of electron temperature (defined only for a subset of the lineouts, since tracing
        is performed in advance)
        """
        return compute_parallel_gradient(self.lineouts[lineout], snaps.electron_temp)

    def ion_temp_parallel_gradient(self, lineout: str, snaps: xr.Dataset):
        """
        Parallel gradient of ion temperature (defined only for a subset of the lineouts, since tracing
        is performed in advance)
        """
        return compute_parallel_gradient(self.lineouts[lineout], snaps.ion_temp)

    def effective_parallel_exb(self, lineout: str, snaps: xr.Dataset):
        """
        The effective parallel ExB velocity (i.e. the parallel velocity that would cause as much poloidal
        transport as the poloidal ExB velocity)
        """
        effective_parallel_exb = effective_parallel_exb_velocity(
            self.grid, self.equi, self.norm, snaps.potential
        )
        return self.lineouts[lineout].interpolate(effective_parallel_exb)

    def q_parallel(self, lineout: str, snaps: xr.Dataset) -> xr.DataArray:
        """Calculates and returns the parallel heat flux"""
        density = self.density(lineout, snaps)
        electron_temp = self.electron_temp(lineout, snaps)
        electron_temp_parallel_gradient = self.electron_temp_parallel_gradient(
            lineout, snaps
        )
        ion_temp = self.ion_temp(lineout, snaps)
        ion_temp_parallel_gradient = self.ion_temp_parallel_gradient(lineout, snaps)
        ion_velocity = self.velocity(lineout, snaps)
        current = self.current(lineout, snaps)
        effective_parallel_exb = self.effective_parallel_exb(lineout, snaps)

        q_par = total_parallel_heat_flux(
            density,
            electron_temp,
            electron_temp_parallel_gradient,
            ion_temp,
            ion_temp_parallel_gradient,
            ion_velocity,
            current,
            effective_parallel_exb,
            self.norm,
        )

        return q_par

    def plot_lineouts(self):
        """Plots the magnetic geometry and the lineouts (as a sanity check)"""

        divertor_ = read_from_json(tcvx21.divertor_coords_json)

        _, ax = plt.subplots(figsize=(10, 10))
        plt.contour(
            self.grid.r_s,
            self.grid.z_s,
            self.equi.normalised_flux_surface_label(self.grid.r_s, self.grid.z_s),
        )

        plt.scatter(
            self.lineouts["xpt"].r_points,
            self.lineouts["xpt"].z_points,
            s=1,
            marker=".",
            color="r",
            label="xpt",
        )
        plt.scatter(
            self.lineouts["rdpa"].r_points,
            self.lineouts["rdpa"].z_points,
            s=1,
            marker="+",
            color="k",
            label="rdpa",
        )

        for key, lineout in self.lineouts.items():

            if key in ["rdpa", "xpt"]:
                continue

            plt.plot(lineout.r_points, lineout.z_points, label=key, linewidth=2.5)

            if hasattr(lineout, "forward_lineout"):
                plt.plot(
                    lineout.forward_lineout.r_points,
                    lineout.forward_lineout.z_points,
                    label=f"{key}+",
                    linewidth=2.5,
                )
            if hasattr(lineout, "reverse_lineout"):
                plt.plot(
                    lineout.reverse_lineout.r_points,
                    lineout.reverse_lineout.z_points,
                    label=f"{key}-",
                    linewidth=2.5,
                )

        plt.legend()
        if self.equi.flipped_z:
            ax.invert_yaxis()
        ax.set_aspect("equal")

        plt.plot(
            divertor_["r_points"] / self.equi.axis_r.values,
            divertor_["z_points"] / self.equi.axis_r.values * -1.0
            if self.equi.flipped_z
            else 1.0,
            color="k",
        )
