"""
Interface to snapshot files
"""
from pathlib import Path
import numpy as np
import xarray as xr
import time

variable_dict = {
    "logne": "density",
    "logte": "electron_temp",
    "logti": "ion_temp",
    "potxx": "potential",
    # "vortx": "vorticity",
    "jparx": "current",
    "uparx": "velocity",
    # "aparx": "apar",
    "tau": "tau",
}

drop_vars = [
    "logne_perpghost",
    "logte_perpghost",
    "logti_perpghost",
    "potxx_perpghost" "vortx",
    "vortx_perpghost",
    "jparx_perpghost",
    "uparx_perpghost" "aparx",
    "aparx_perpghost",
    "tau",
]
only_tau = [
    "logne",
    "logne_perpghost",
    "logte",
    "logte_perpghost",
    "logti",
    "logti_perpghost",
    "potxx",
    "potxx_perpghost",
    "vortx",
    "vortx_perpghost",
    "jparx",
    "jparx_perpghost",
    "uparx",
    "uparx_perpghost",
    "aparx",
    "aparx_perpghost",
]


def get_snap_length_from_file(file_path: Path):
    return xr.open_dataset(file_path / "snaps00000.nc").sizes["dim_tau"]


def read_snaps_from_file(
    file_path: Path,
    norm,
    time_slice: slice = slice(-1, None),
    all_planes: bool = False,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Reads a single snapshot file (corresponding to a single toroidal plane)

    A subset of the time-interval can be loaded to reduce the memory requirements
    """

    if all_planes:
        if verbose:
            print("Reading all planes")

        snaps = []
        for plane in range(len([snap for snap in file_path.glob("snaps*.nc")])):

            if verbose:
                print(f"Reading plane {plane}", end=". ")
            start = time.time()

            snap = xr.open_dataset(
                file_path / f"snaps{plane:05d}.nc",
                chunks={"dim_tau": 50},
                engine="netcdf4",
                drop_variables=drop_vars,
            )
            snaps.append(
                snap.isel(dim_tau=time_slice).persist().expand_dims({"phi": [plane]})
            )
            snap.close()

            if verbose:
                print(f"Finished in {time.time() - start:4.3f}s")

        if verbose:
            print(f"Concatenating planes", end=". ")
        start = time.time()
        dataset = xr.concat(snaps, dim="phi")
        if verbose:
            print(f"Finished in {time.time() - start:4.3f}s")

    else:
        if verbose:
            print("Reading one plane")
        dataset = xr.open_dataset(
            str(file_path / "snaps00000.nc"),
            chunks={"dim_tau": 50},
            drop_variables=drop_vars,
        ).expand_dims({"phi": [0]})
        dataset = dataset.isel(dim_tau=time_slice)
        dataset.close()

    tau = xr.open_dataset(file_path / "snaps00000.nc", drop_variables=only_tau)[
        "tau"
    ].isel(dim_tau=time_slice)
    tau.close()

    dataset["tau"] = tau
    dataset = dataset.rename({"dim_tau": "tau"})

    normalisation = {
        "tau": norm.tau_0,
        "density": norm.n0,
        "electron_temp": norm.Te0,
        "ion_temp": norm.Ti0,
        "velocity": norm.c_s0,
        "current": (norm.c_s0 * norm.elementary_charge * norm.n0).to(
            "kiloampere*meter**-2"
        ),
        "potential": (norm.Te0 / norm.elementary_charge).to("kilovolt"),
        "vorticity": (
            norm.Mi
            * norm.n0
            * norm.Te0
            / (norm.elementary_charge * norm.rho_s0 ** 2 * norm.B0 ** 2)
        ).to("coulomb/meter**3"),
        "apar": norm.beta_0 * norm.B0 * norm.rho_s0,
    }

    snaps = xr.Dataset()

    for var_access, variable in variable_dict.items():
        attrs = {"name": variable, "norm": normalisation[variable]}

        if variable == "tau":
            snaps[variable] = dataset[variable].assign_attrs(**attrs)
        else:
            var = dataset[var_access].rename({"dim_vgrid": "points"})

            if dataset[var_access].logquantity:
                snaps[variable] = np.exp(var).assign_attrs(**attrs)
            else:
                snaps[variable] = var.assign_attrs(**attrs)

    return snaps
