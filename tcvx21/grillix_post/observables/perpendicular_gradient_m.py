import numpy as np
import xarray as xr
from tcvx21.grillix_post.components import poloidal_vector


def perpendicular_gradient(
    grid, norm, shaped_array: xr.DataArray, normalised_to_Lperp: bool = True
):
    """
    Takes the perpendicular gradient of a shaped array (i.e. already has had vector_to_matrix applied)

    Returns as a vector array.

    If normalised_to_Lperp, returns an array which is the gradient in units of 1/rho_s0 (perpendicular length scale)
    Otherwise, returns an array which is the gradient in units of 1/R0 (parallel length scale)
    """

    dims = shaped_array.dims
    assert (
        "R" in dims and "Z" in dims
    ), f"Should pass shaped array (i.e. apply vector_to_matrix) to perpendicular_gradient"

    delta = (norm.R0 / norm.rho_s0).to("").magnitude
    if normalised_to_Lperp:
        # Take the gradient w.r.t. perpendicular length scale
        gradient = np.gradient(
            shaped_array, grid.spacing * delta, axis=(dims.index("R"), dims.index("Z"))
        )
        gradient_scale_length = norm.rho_s0
    else:
        # Take the gradient w.r.t. parallel length scale
        gradient = np.gradient(
            shaped_array, grid.spacing, axis=(dims.index("R"), dims.index("Z"))
        )
        gradient_scale_length = norm.R0

    xr_data = {
        "dims": shaped_array.dims,
        "coords": shaped_array.coords,
        "attrs": shaped_array.attrs,
    }
    vector_grad = poloidal_vector(input_r=gradient[0], input_z=gradient[1], **xr_data)

    if "norm" in vector_grad.attrs:
        assert not isinstance(
            vector_grad.norm, str
        ), f"Cannot operate on units in string format (in perpendicular_gradient). Norm was\
            {vector_grad.norm}. Convert to a Quantity via the Normalisation before calling this function."
        vector_grad.attrs["norm"] *= 1.0 / gradient_scale_length
    else:
        vector_grad.attrs["norm"] = 1.0 / gradient_scale_length

    return vector_grad
