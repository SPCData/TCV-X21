"""
Implementations of lineouts specific to the tcvx21 analysis
"""
import numpy as np
import warnings
from pathlib import Path
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import xarray as xr
import pint

from .lineout_m import Lineout, on_grid_interpolator
from tcvx21.analysis.contour_finding_m import find_contours
from tcvx21.file_io.json_io_m import read_from_json
from tcvx21.file_io.matlab_file_reader_m import read_struct_from_file
from tcvx21.units_m import Quantity, convert_xarray_to_quantity


def outboard_midplane_chord(grid, equi, n_points: int = 1000):
    """
    Returns n_points of equally spaced points along the outboard zaxis, which is a horizontal line from the magnetic axis
    outwards (to large radius). N.b. the outboard midplane is usually to largest radius, but this is roughly equivalent
    """

    r_start = equi.axis_r_norm
    z_start = equi.axis_z_norm

    r_end = grid.r_s.max()
    z_end = equi.axis_z_norm

    lineout = Lineout([r_start, r_end], [z_start, z_end])
    lineout.find_points_on_grid(grid, n_points=n_points, trim_points=False)

    return lineout


def penalisation_contour(
    grid,
    equi,
    level: float,
    contour_index: int = None,
    epsilon: float = 1e-10,
    smoothing: float = 0.1,
    n_points: int = 1000,
):
    """
    Returns a lineout along a contour of the penalisation characteristic

    Should use torx.specialisations.*.penalisation_m.filled_penalisation_characteristic to get the
    penalisation_characteristic filled to include pghost points

    smoothing is recommended, since this removes the grid staircase pattern of the contour
    """
    penalisation_characteristic = grid.shape(equi.penalisation_characteristic)

    assert (
        "R" in penalisation_characteristic.dims
        and "Z" in penalisation_characteristic.dims
    ), "Call vector_to_matrix on\
        penalisation_chacteristic before calling penalisation_contour"

    # Shift the level slightly in from [0, 1] to ensure a clear contour
    level = max(level, epsilon)
    level = min(level, 1.0 - epsilon)

    contours = find_contours(
        grid.r_s, grid.z_s, penalisation_characteristic, level=level
    )

    if equi.flipped_z:
        # Reverse the contour list is using flipped_z, since contours are returned from bottom to top
        contours.reverse()

    if contour_index is None:  # pragma: no cover
        plt.pcolormesh(
            grid.r_s, grid.z_s, penalisation_characteristic, shading="nearest"
        )
        for i, contour in enumerate(contours):
            r_points, z_points = contour.T

            plt.plot(r_points, z_points, label=f"contour_index = {i}")
        plt.legend()
        plt.show()

        raise ValueError("Select a contour index for penalisation_contour")

    else:
        r_points, z_points = contours[contour_index].T

    lineout = Lineout(r_points, z_points, smoothing=smoothing)
    lineout.find_points_on_grid(grid, trim_points=True, n_points=n_points)

    return lineout


def thomson_scattering(grid, equi, thomson_position: Path, n_points: int = 1000):
    """
    Returns a lineout for the Thomson scattering points below the magnetic axis
    """
    thomson_position_ = read_from_json(thomson_position)

    # For the simulation, we only want divertor entrance points (below the magnetic axis)
    # Use the ts_below_axis mask to limit to this region
    ts_below_axis = thomson_position_["Z"] < 0
    thomson_position_ = {
        "R": thomson_position_["R"][ts_below_axis],
        "Z": thomson_position_["Z"][ts_below_axis],
    }

    R0 = convert_xarray_to_quantity(equi.axis_r)

    ts_r = (Quantity(thomson_position_["R"], "m") / R0).to("").magnitude
    ts_z = (Quantity(thomson_position_["Z"], "m") / R0).to("").magnitude * (
        -1.0 if equi.flipped_z else 1.0
    )

    ts = Lineout(ts_r, ts_z)
    ts.find_points_on_grid(grid, n_points=n_points, trim_points=False)

    return ts


def rdpa(grid, equi, omp_map, norm, rdpa_coordinates: Path):
    """
    Returns a lineout for the 2D RDPA (reciprocating divertor probe array). Despite being a 2D diagnostic,
    since observables are not on a regular rectangular grid we calculate values as tricolumn (R^u-R^_sep, Z, values)
    data and then reshape.
    """
    warnings.simplefilter("ignore", category=pint.errors.UnitStrippedWarning)

    rdpa_coords = read_from_json(rdpa_coordinates)

    R0 = norm.R0

    rdpa_r = xr.DataArray(
        Quantity(rdpa_coords["R"], rdpa_coords["R_units"]) / R0
    ).assign_attrs(norm=R0)
    rdpa_zx = xr.DataArray(
        Quantity(rdpa_coords["Zx"], rdpa_coords["Zx_units"]) / R0
    ).assign_attrs(norm=R0)
    rdpa_rsep = xr.DataArray(
        Quantity(rdpa_coords["Rsep"], rdpa_coords["Rsep_units"]) / R0
    ).assign_attrs(norm=R0)
    rdpa_z = xr.DataArray(
        Quantity(rdpa_coords["Z"], rdpa_coords["Z_units"]) / R0
    ).assign_attrs(norm=R0)

    if equi.flipped_z:
        rdpa_z *= -1

    # Some points cannot be matched in the reference equilibrium. These are marked by NaN
    nan_value = ~np.isnan(rdpa_r).values
    rdpa_on_grid = on_grid_interpolator(grid)(rdpa_r, rdpa_z, grid=False) > 0.99
    mask = np.logical_and.reduce((nan_value, rdpa_on_grid))
    print(f"Masking {np.count_nonzero(~mask)} of {rdpa_r.size} RDPA points")
    rdpa_r, rdpa_z, rdpa_zx, rdpa_rsep = (
        rdpa_r[mask],
        rdpa_z[mask],
        rdpa_zx[mask],
        rdpa_rsep[mask],
    )

    rdpa_rho = equi.normalised_flux_surface_label(rdpa_r, rdpa_z, grid=False)
    assert np.allclose(
        rdpa_rsep, omp_map.convert_rho_to_distance(rdpa_rho), atol=1e-6
    ), f"Max difference for rdpa Rsep was {np.max(np.abs(rdpa_rsep - omp_map.convert_rho_to_distance(rdpa_rho)))}"

    rdpa_lineout = Lineout(rdpa_r, rdpa_z)
    rdpa_lineout.setup_interpolation_matrix(grid, use_source_points=True)

    # Store the observable positions to allow mapping to 2D data
    # We store according to machine coordinates, regardless of whether we have flipped the equilibrium
    rdpa_lineout.coords = {
        "R": convert_xarray_to_quantity(rdpa_r),
        "Rsep": convert_xarray_to_quantity(rdpa_rsep),
        "Z": convert_xarray_to_quantity(rdpa_z) * (-1.0 if equi.flipped_z else 1.0),
        "Zx": convert_xarray_to_quantity(rdpa_zx),
    }

    return rdpa_lineout


def xpoint(
    grid,
    equi,
    norm,
    r_range: float = 0.1,
    z_range: float = 0.1,
    rhomin=0.97,
    rhomax=1.03,
    r_samples: int = 50,
    z_samples: int = 50,
):
    """
    Returns a lineout of the region around the X-point
    """
    warnings.simplefilter("ignore", category=pint.errors.UnitStrippedWarning)

    r_min = float(equi.x_point_r_norm.values - r_range)
    r_max = float(equi.x_point_r_norm.values + r_range)
    z_min = float(equi.x_point_z_norm.values - z_range)
    z_max = float(equi.x_point_z_norm.values + z_range)

    r_sample, z_sample = np.linspace(r_min, r_max, num=r_samples), np.linspace(
        z_min, z_max, num=z_samples
    )

    xpoint_on_grid = (
        on_grid_interpolator(grid)(r_sample, z_sample, grid=True) > 0.99
    ).T
    xpoint_rho = equi.normalised_flux_surface_label(
        r_sample, z_sample, grid=True
    ).values

    r_mesh, z_mesh = np.meshgrid(r_sample, z_sample)
    mask = np.logical_and.reduce(
        (xpoint_on_grid, xpoint_rho > rhomin, xpoint_rho < rhomax)
    )

    xpoint_lineout = Lineout(r_mesh[mask].ravel(), z_mesh[mask].ravel())
    xpoint_lineout.setup_interpolation_matrix(grid, use_source_points=True)

    xpoint_lineout.coords = {
        "R": xpoint_lineout.r_points * norm.R0,
        "Z": xpoint_lineout.z_points * norm.R0 * (-1.0 if equi.flipped_z else 1.0),
        "Zx": xpoint_lineout.z_points * norm.R0 * (-1.0 if equi.flipped_z else 1.0)
        - convert_xarray_to_quantity(equi.axis_r),
    }

    return xpoint_lineout
