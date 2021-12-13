"""
Interface to the simulation grid
"""
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

district_dict = {
    -1000: "OFF_GRID",
    # point off the grid (i.e. normally 'nan')
    813: "CORE",
    # point located in core (outside actual computational domain, rho<rhomin)
    814: "CLOSED",
    # point located in closed field line region (within computational domain)
    815: "SOL",
    # point located in scrape-off layer (within computational domain)
    816: "PRIVFLUX",
    # point located in private flux region (within computational domain)
    817: "WALL",
    # point located in wall (outside computational domain, rho>rhomax)
    818: "DOME",
    # point located in divertor dome (outside computational domain, e.g. rho<rhomin_privflux)
    819: "OUT",
    # point located outside additional masks, i.e. shadow region (outside computational domain)
}


def calculate_shaping(r_u, z_u):
    """
    Calculates the constant shaping arrays
    """
    r_s, z_s = np.unique(r_u), np.unique(z_u)

    point_index = xr.DataArray(np.arange(r_u.size), dims="points")
    tricolumn_data = np.column_stack((r_u, z_u, point_index))

    shaped_index = pd.DataFrame(tricolumn_data, columns=["x", "y", "z"]).pivot_table(
        values="z", index="y", columns="x", dropna=False
    )

    mask = xr.DataArray(
        np.where(np.isnan(shaped_index), np.nan, 0),
        dims=["Z", "R"],
        coords={"Z": z_s, "R": r_s},
    )

    shaped_index = xr.DataArray(
        np.nan_to_num(shaped_index.to_numpy(), 0).astype(int),
        dims=["Z", "R"],
        coords={"Z": z_s, "R": r_s},
    )

    return mask, shaped_index, point_index


def shape_single(r_u, z_u, input_array):
    """Shapes a single array"""
    mask, shaped_index, point_index = calculate_shaping(r_u, z_u)

    return input_array.isel(points=shaped_index) + mask


class Grid:
    def __init__(self, grid_file: Path):
        """
        Initialises the grid object from a grid file
        """
        assert grid_file.exists()

        grid_dataset = xr.open_dataset(grid_file)

        r_u, z_u, self.spacing = self.get_unstructured_grid_arrays_from_file(
            grid_dataset
        )
        r_s, z_s = np.unique(r_u), np.unique(z_u)
        self.size = r_u.size
        self.grid_size = grid_dataset.sizes["dim_nl"]
        self.r_u, self.z_u, self.r_s, self.z_s = r_u, z_u, r_s, z_s

        mask, shaped_index, point_index = calculate_shaping(r_u, z_u)

        self.point_index, self.shaped_index, self.mask = point_index, shaped_index, mask

        # Read and shape the districts array
        raw_districts = grid_dataset["info"][0, :]
        raw_districts = self.shape(raw_districts.rename({"dim_nl": "points"}))

        districts = np.empty_like(raw_districts).astype(str)
        for key, value in district_dict.items():
            districts[
                np.nan_to_num(raw_districts.values, nan=-1000).astype(int) == key
            ] = value

        self.districts = xr.DataArray(
            districts, dims=("Z", "R"), coords={"Z": z_s, "R": r_s}
        )

    @staticmethod
    def get_unstructured_grid_arrays_from_file(grid_ds: xr.Dataset):
        """Given a netcdf grid file, return the x_unstructured and y_unstructred arrays"""

        x_unstructured = grid_ds.xmin + (grid_ds["li"].values - 1) * grid_ds.hf
        y_unstructured = grid_ds.ymin + (grid_ds["lj"].values - 1) * grid_ds.hf

        spacing = np.mean(np.diff(np.unique(x_unstructured)))
        assert np.allclose(
            np.diff(np.unique(x_unstructured)), spacing
        ), "Error: x basis vector is not equally spaced."
        assert np.allclose(
            np.diff(np.unique(y_unstructured)), spacing
        ), "Error: x basis vector is not equally spaced."

        return x_unstructured, y_unstructured, spacing

    def shape(self, input_array: xr.DataArray):
        """
        Converts from tricolumn data to a 2D matrix
        """

        return input_array.isel(points=self.shaped_index) + self.mask

    def flatten(self, input_array: xr.DataArray):
        """
        Converts from a 2D matrix to tricolumn data
        """

        def flatten_2D(array):
            return array.flatten()[np.logical_not(self.mask.values.flatten())]

        return xr.apply_ufunc(
            flatten_2D,
            input_array,
            input_core_dims=[
                ["Z", "R"],
            ],
            output_core_dims=[["points"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"points": self.size}},
            output_dtypes=[np.float64],
        )
