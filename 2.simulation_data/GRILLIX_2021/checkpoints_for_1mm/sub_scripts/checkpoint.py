#!/usr/bin/env python3

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from shutil import copy2


def crop_snap(in_file, out_file, tau_slice: slice, reset_tau=False):

    new_dataset = xr.open_dataset(in_file)
    new_dataset.encoding = {"unlimited_dims": ["dim_tau"]}

    new_dataset = new_dataset.isel(dim_tau=tau_slice, drop=False)

    if reset_tau:
        new_dataset["tau"] = new_dataset["tau"] - new_dataset["tau"][-1]

    new_dataset.attrs["nsnaps_last"] = new_dataset.sizes["dim_tau"]

    out_file.parent.mkdir(exist_ok=True)
    new_dataset.to_netcdf(out_file)


def crop_diagnostics_scalar(in_file, out_file, tau_slice: slice, reset_tau=False):

    new_dataset = xr.open_dataset(in_file)
    new_dataset.encoding = {"unlimited_dims": ["dim_time"]}

    new_dataset = new_dataset.isel(dim_time=tau_slice, drop=False)
    new_dataset.attrs["ndiag_last"] = new_dataset.sizes["dim_time"]

    if reset_tau:
        new_dataset["diags"][:, 0] = (
            new_dataset["diags"][:, 0] - new_dataset["diags"][-1, 0]
        )

    new_dataset.attrs["tau_last"] = np.max(new_dataset.diags[:, 0].values)

    out_file.parent.mkdir(exist_ok=True)
    new_dataset.to_netcdf(out_file)


def crop_diagnostics_zonal(in_file, out_file, tau_slice: slice, reset_tau=False):

    new_dataset = xr.open_dataset(in_file)
    new_dataset.encoding = {"unlimited_dims": ["dim_time"]}

    new_dataset = new_dataset.isel(dim_time=tau_slice, drop=False)
    new_dataset.attrs["ndiag_last"] = new_dataset.sizes["dim_time"]

    if reset_tau:
        new_dataset["tau"] = new_dataset["tau"] - new_dataset["tau"][-1]

    new_dataset.attrs["tau_last"] = np.max(new_dataset["tau"].values)

    out_file.parent.mkdir(exist_ok=True)
    new_dataset.to_netcdf(out_file)


run_dir = Path("").absolute()
print(f"Running in {run_dir}")
dt_string = datetime.now().strftime("%d_%b_%Y_%H:%M:%S")
print("date and time =", dt_string)

tau_slice = slice(-5, None)

checkpoint_dir = run_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

output_dir = checkpoint_dir / dt_string
output_dir.mkdir(exist_ok=False)

print("Copying parameter file")
copy2(run_dir / "params.in", output_dir / "params.in")

for snap_file in (run_dir).glob("snaps*.nc"):
    print(f"Processing {snap_file.name}")

    out_file = output_dir / snap_file.name

    crop_snap(snap_file, out_file, tau_slice=tau_slice)

print(f"Processing diagnostics")
crop_diagnostics_scalar(
    run_dir / "diagnostics_scalar.nc",
    output_dir / "diagnostics_scalar.nc",
    tau_slice=tau_slice,
)
crop_diagnostics_zonal(
    run_dir / "diagnostics_zonal.nc",
    output_dir / "diagnostics_zonal.nc",
    tau_slice=tau_slice,
)

print("Done")
