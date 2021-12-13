import xarray as xr
import numpy as np
from pathlib import Path
from tcvx21.grillix_post.components import FieldlineTracer
from tcvx21.grillix_post.lineouts import Lineout

xr.set_options(keep_attrs=True)


def initialise_lineout_for_parallel_gradient(
    lineout, grid, equi, norm, npol, stored_trace: Path = None
):
    """
    Traces to find the forward and reverse lineouts for a given lineout

    Expensive! Needs to be done once per lineout that you want to take gradients with
    """
    fieldline_tracer = FieldlineTracer(equi)

    try:
        print(f"Attempting to read stored trace from {stored_trace}")
        ds = xr.open_dataset(stored_trace)
        assert np.allclose(ds["lineout_x"], lineout.r_points)
        assert np.allclose(ds["lineout_y"], lineout.z_points)
    except (FileNotFoundError, ValueError):
        forward_trace, reverse_trace = fieldline_tracer.find_neighbouring_points(
            lineout.r_points, lineout.z_points, n_toroidal_planes=int(npol)
        )

        ds = xr.Dataset(
            data_vars=dict(
                forward_x=("points", forward_trace[:, 0]),
                forward_y=("points", forward_trace[:, 1]),
                forward_l=("points", forward_trace[:, 2]),
                reverse_x=("points", reverse_trace[:, 0]),
                reverse_y=("points", reverse_trace[:, 1]),
                reverse_l=("points", reverse_trace[:, 2]),
                lineout_x=("points", lineout.r_points),
                lineout_y=("points", lineout.z_points),
            )
        )

        if stored_trace is not None:
            if stored_trace.exists():
                stored_trace.unlink()
            ds.to_netcdf(stored_trace)

    lineout.forward_lineout = Lineout(ds["forward_x"], ds["forward_y"])
    lineout.forward_lineout.setup_interpolation_matrix(grid, use_source_points=True)
    lineout.reverse_lineout = Lineout(ds["reverse_x"], ds["reverse_y"])
    lineout.reverse_lineout.setup_interpolation_matrix(grid, use_source_points=True)

    lineout.forward_distance = xr.DataArray(
        ds["forward_l"], dims="interp_points"
    ).assign_attrs(norm=norm.R0)
    lineout.reverse_distance = xr.DataArray(
        ds["reverse_l"], dims="interp_points"
    ).assign_attrs(norm=norm.R0)


def compute_parallel_gradient(lineout, field):
    """
    Computes the parallel gradient via centred differences

    Note that you should multiply this by the penalisation direction function to get the direction 'towards the
    wall'. This isn't quite the same as projecting onto the wall normal, but for computing the parallel
    heat flux this is actually more helpful
    """

    assert hasattr(lineout, "forward_lineout") and hasattr(
        lineout, "reverse_lineout"
    ), f"Have to call initialise_lineout_for_parallel_gradient on lineout before trying to compute_parallel_gradient"

    parallel_gradients = [
        compute_gradient_on_plane(lineout, field, plane)
        for plane in range(field.sizes["phi"])
    ]

    return xr.concat(parallel_gradients, dim="phi")


def compute_gradient_on_plane(lineout, field, plane):
    """Computes the parallel gradient on a single plane"""

    forward_value = lineout.forward_lineout.interpolate(
        field.isel(phi=np.mod(plane + 1, field.sizes["phi"]))
    )
    reverse_value = lineout.forward_lineout.interpolate(
        field.isel(phi=np.mod(plane - 1, field.sizes["phi"]))
    )

    two_plane_distance = lineout.forward_distance - lineout.reverse_distance

    centred_difference = forward_value - reverse_value
    return (
        (centred_difference / two_plane_distance)
        .assign_coords(phi=plane)
        .assign_attrs(norm=field.norm / two_plane_distance.norm)
    )
