"""
Defines an implementation for lineouts based on curves in the poloidal plane, which are typically close to orthogonal
to magnetic flux surfaces
"""
import numpy as np
import xarray as xr

from scipy.interpolate.fitpack2 import RectBivariateSpline
from .csr_linear_interpolation_m import make_matrix_interp
from tcvx21.units_m import Quantity
from .parametric_spline_m import ParametricSpline


def on_grid_interpolator(grid) -> RectBivariateSpline:
    """
    Returns an interpolator which returns 1.0 if an arbitrary (R, Z) position is on the grid, and 0.0 if not
    """
    on_grid = np.nan_to_num(
        grid.shape(xr.DataArray(np.ones_like(grid.r_u), dims="points")), nan=0.0
    )
    return RectBivariateSpline(grid.r_s, grid.z_s, on_grid.T, kx=1, ky=1)


class Lineout:
    """
    A "lineout" which allows you to map from grid points to an arbitrary (R, Z) poloidal line or curve
    """

    def __init__(
        self,
        r_points: np.ndarray,
        z_points: np.ndarray,
        periodic: bool = False,
        smoothing: float = 0.0,
    ):
        """
        Initialises the Lineout object from a specified set of r and z points

        If smoothing if > 0.0, then the input points will be smoothed. (Typically smoothing required << 1, recommend
        to check the fit)
        """
        r_points, z_points = np.array(r_points), np.array(z_points)

        assert r_points.ndim == 1
        assert r_points.shape == z_points.shape
        self.r_source, self.z_source = r_points, z_points

        # Default use a third order spline, unless not enough points are given in which case drop the order
        order = min(3, r_points.size - 1)

        self.curve = ParametricSpline(
            sample_points=[r_points, z_points],
            periodic=periodic,
            smoothing=smoothing,
            order=order,
        )

    def find_points_on_grid(
        self,
        grid,
        n_points,
        test_points: int = 1000,
        epsilon: float = 1e-6,
        trim_points: bool = False,
    ):
        """
        Finds 'n_points' equally spaced (R, Z) points along the lineout

        Note that 'curve' must have a single segment on the grid -- this algorithm will fail if there is an off grid
        segment between two on-grid segments

        n_points * precision test points will be used to check which values of the parameter are on the grid
        epsilon shifts the endpoints slightly inwards to ensure that the returned points are actually on the grid
        """
        on_grid = on_grid_interpolator(grid)

        t_tests = np.linspace(
            self.curve.t_min, self.curve.t_max, num=test_points, endpoint=True
        )

        test_on_grid = on_grid(*self.curve(t_tests), grid=False)
        t_on_grid = t_tests[np.where(test_on_grid > 0.99)]

        # Find the endpoints
        t_min, t_max = np.min(t_on_grid), np.max(t_on_grid)

        if trim_points:
            # Trim the lineout to exclude the first 5% and last 5% of the data, since we shouldn't take statistics in
            # the boundary conditions (pinned values have extreme skew and kurtosis)
            t_min += 0.1 * (t_max - t_min)
            t_max -= 0.1 * (t_max - t_min)

        # Find the points inside the interval
        t_points = np.linspace(t_min + epsilon, t_max - epsilon, num=n_points)
        r_points, z_points = self.curve(t_points)

        # Make sure that all of the returned points are on the grid. min should be close to 1.0, but allow edge points
        assert (
            np.min(on_grid(r_points, z_points, grid=False)) > 0.5
        ), f"get_n_points_on_grid returned >1 off-grid point.\
            Check that the lineout has a single segment on the grid"

        self.r_points, self.z_points = r_points, z_points
        self.setup_interpolation_matrix(grid)

    def setup_interpolation_matrix(self, grid, use_source_points: bool = False):
        """
        Adds attributes required for interpolation
        """
        if use_source_points:
            self.r_points = self.r_source
            self.z_points = self.z_source

        self.interpolation_matrix = make_matrix_interp(
            grid.r_u, grid.z_u, self.r_points, self.z_points
        )
        self.grid_size, self.curve_size = grid.size, self.r_points.size

    def interpolate_1d(self, input_array: np.ndarray):
        """
        Applies the interpolation matrix to an array with dimension 'points'
        """
        return self.interpolation_matrix * input_array

    def interpolate(self, input_array: xr.DataArray):
        """
        Uses a dask-parallelized algorithm to apply the csr-matrix interpolation over the input array

        See http://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html for hints on writing
        ufuncs for xarrays
        """

        interpolated = xr.apply_ufunc(
            # First pass the function
            self.interpolate_1d,
            # Next, pass the input array, and chunk it in such a way that dask can neatly parallelize it
            input_array,
            # Provide a list of the 'core' dimensions, which are the ones that aren't looped over
            input_core_dims=[
                ["points"],
            ],
            # Provide a list of the output dimensions (which replaces input_core_dims)
            output_core_dims=[["interp_points"]],
            # Set which dimensions are allowed to change size (doesn't seem to have any effect)
            # exclude_dims=set(("points",)),
            # Loop over non-core dims
            vectorize=True,
            # Pass keywords to the dask core, to define the size of the returned array
            # Switch on dask parallelism. Works best if input_array is generated using the dask-based methods
            dask="parallelized",
            # Declare the size of the output_core_dims
            dask_gufunc_kwargs={"output_sizes": {"interp_points": self.curve_size}},
            # Declare the type of the output
            output_dtypes=[np.float64],
        )

        interpolated.attrs = input_array.attrs

        return interpolated

    def poloidal_arc_length(self, norm) -> xr.DataArray:
        """
        Calculates the poloidal arc length of a list of (R, Z) points by using simple finite difference

        The result is the same length as the input array, due to a zero-insertion for the first point
        """
        return xr.DataArray(
            np.cumsum(
                np.insert(
                    np.sqrt(np.diff(self.r_points) ** 2 + np.diff(self.z_points) ** 2),
                    0,
                    0.0,
                )
            )
        ).assign_attrs(
            norm=("R0" if norm is None else norm["R0"]), name="poloidal arc length"
        )

    def omp_mapped_distance(self, omp_map) -> Quantity:
        """
        Calculates the OMP-mapped radial distance
        """
        values = omp_map(self.r_points, self.z_points)
        return values.values * values.norm
