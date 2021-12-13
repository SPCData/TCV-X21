"""
Defines routines for calculating integral quantities

Automatically handles units
"""
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from tcvx21.units_m import Quantity


def poloidal_integration(grid, norm, poloidal_function: xr.DataArray):
    """
    Returns the 2D poloidal integral of a function defined

    Off-grid values are assigned a value of 0.0, and a filled grid spline interpolator is used to evaluate the integral
    """
    assert hasattr(poloidal_function, "norm")
    assert isinstance(poloidal_function.norm, Quantity)

    grid_limits = {
        "xa": grid.r_s.min(),
        "xb": grid.r_s.max(),
        "ya": grid.z_s.min(),
        "yb": grid.z_s.max(),
    }

    poloidal_function_values = np.nan_to_num(poloidal_function.values, nan=0.0)

    function_integral = RectBivariateSpline(
        grid.r_s, grid.z_s, poloidal_function_values.T
    ).integral(**grid_limits)

    norm = poloidal_function.norm * norm.R0 ** 2

    return function_integral * norm


def axisymmetric_cylindrical_integration(grid, norm, poloidal_function: xr.DataArray):
    """
    Returns the 3D cylindrically-integrated value of a function defined over the poloidal grid

    N.b. automatically takes care of the "R" factor introduced by the \theta integral
    """

    r_s_grid, _ = np.meshgrid(grid.r_s, grid.z_s)
    poloidal_integration_function = xr.DataArray(
        poloidal_function * r_s_grid
    ).assign_attrs(norm=poloidal_function.norm)
    function_integral = (
        2.0
        * np.pi
        * norm.R0
        * poloidal_integration(grid, norm, poloidal_integration_function)
    )

    return function_integral
