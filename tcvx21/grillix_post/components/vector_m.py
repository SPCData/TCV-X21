"""
Cylindrical vector operations based on xarray labelled axes
"""
import xarray as xr
import numpy as np


def _add_vector_dim(dims, coords):
    """Adds a 'vector' dimension with coords (eR, ePhi, eZ)"""
    dims = list(dims)
    dims.append("vector")

    coords = dict(coords)
    coords["vector"] = ["eR", "ePhi", "eZ"]

    return dims, coords


def toroidal_vector(input_array: xr.DataArray):
    """
    Builds a cylindrical vector-field array from a phi component array, and zeros the (R, Z) components
    """
    vector_array = np.zeros(tuple(list(input_array.shape) + [3]))
    vector_array[..., 1] = input_array

    dims, coords = _add_vector_dim(input_array.dims, input_array.coords)

    return xr.DataArray(vector_array, dims=dims, attrs=input_array.attrs, coords=coords)


def poloidal_vector(
    input_r: xr.DataArray,
    input_z: xr.DataArray,
    dims: list,
    coords: dict = {},
    attrs: dict = {},
):
    """
    Builds a cylindrical vector-field array from (R, Z) component arrays, and zeros the phi component

    dims should be the names of the dimensions (i.e. from input_array.dims)
    Optional: coords and attrs are used to set the coords and attributes of the output array
    """
    assert (
        input_r.shape == input_z.shape
    ), f"input_r and input_z shapes are {input_r.shape} and {input_z.shape}.\
        Please broadcast shapes to match before passing to poloidal vector"

    vector_array = np.zeros(tuple(list(input_r.shape) + [3]))
    vector_array[..., 0] = input_r
    vector_array[..., 2] = input_z

    dims, coords = _add_vector_dim(dims, coords)

    return xr.DataArray(vector_array, dims=dims, attrs=attrs, coords=coords)


def cylindrical_vector(
    input_r: xr.DataArray,
    input_phi: xr.DataArray,
    input_z: xr.DataArray,
    dims: list,
    coords: dict = {},
    attrs: dict = {},
):
    """
    Builds a cylindrical vector-field array from (R, phi, Z) component arrays

    dims should be the names of the dimensions (i.e. from input_array.dims)
    Optional: coords and attrs are used to set the coords and attributes of the output array
    """
    assert (
        input_r.shape == input_phi.shape and input_r.shape == input_z.shape
    ), f"input_r, input_phi and input_z shapes are {input_r.shape}, {input_phi.shape} {input_z.shape}.\
            Please broadcast shapes to match before passing to cylindrical vector"

    vector_array = np.zeros(tuple(list(input_r.shape) + [3]))
    vector_array[..., 0] = input_r
    vector_array[..., 1] = input_phi
    vector_array[..., 2] = input_z

    dims, coords = _add_vector_dim(dims, coords)

    return xr.DataArray(vector_array, dims=dims, attrs=attrs, coords=coords)


def eR_unit_vector(r_norm, z_norm, grid: bool = False):
    """
    Unit vector pointing in the R (radial) direction
    """

    if not grid:
        return poloidal_vector(
            input_r=xr.DataArray(1.0), input_z=xr.DataArray(0.0), dims=[]
        )
    else:
        return poloidal_vector(
            input_r=xr.DataArray(np.ones((z_norm.size, r_norm.size))),
            input_z=xr.DataArray(np.zeros((z_norm.size, r_norm.size))),
            dims=["Z", "R"],
            coords={"R": r_norm, "Z": z_norm},
        )


def ePhi_unit_vector(r_norm, z_norm, grid: bool = False):
    """
    Unit vector pointing in the phi (toroidal) direction
    """

    if not grid:
        return toroidal_vector(xr.DataArray(1.0))
    else:
        return toroidal_vector(xr.DataArray(np.ones((z_norm.size, r_norm.size))))


def eZ_unit_vector(r_norm, z_norm, grid: bool = False):
    """
    Unit vector pointing in the Z (vertical) direction
    """

    if not grid:
        return poloidal_vector(
            input_r=xr.DataArray(0.0), input_z=xr.DataArray(1.0), dims=[]
        )
    else:
        return poloidal_vector(
            input_r=xr.DataArray(np.zeros((z_norm.size, r_norm.size))),
            input_z=xr.DataArray(np.ones((z_norm.size, r_norm.size))),
            dims=["Z", "R"],
            coords={"R": r_norm, "Z": z_norm},
        )


def vector_magnitude(input_array: xr.DataArray):
    """
    Returns the vector magnitude of input_array

    3.39 ms ± 276 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    return np.sqrt(vector_dot(input_array, input_array))


def poloidal_vector_magnitude(input_array: xr.DataArray):
    """
    Returns the vector magnitude of the poloidal component of the input array
    """
    return vector_magnitude(input_array.isel(vector=[0, 2]))


def vector_dot(input_a: xr.DataArray, input_b: xr.DataArray):
    """
    Returns the dot product of the input arrays

    vec(a) dot vec(b) = mag(a) mag(b) cos(theta)
    where theta is the angle between vec(a) and vec(b)

    2.71 ms ± 149 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    return input_a.dot(input_b, dims=["vector"])


def vector_cross(input_a, input_b):
    """
    Returns the cross product of the input arrays

    vec(a) cross vec(b) = mag(a) mag(b) sin(theta) vec(n)
    where theta is the angle between vec(a) and vec(b) and
    where vec(n) is a unit vector perpendicular to both vec(a) and vec(b)
    The sign of vec(b) is given by right-hand-rule (i.e. if 'a' is index finger, 'b' is middle finger, 'n' is thumb, or
    alternatively if 'a' is thumb, 'b' is index finger, 'n' is middle finger)

    Compatible with dask, parallelization uses input_a.dtype as output_dtype
    Taken from https://github.com/pydata/xarray/issues/3279
    """
    return xr.apply_ufunc(
        np.cross,
        input_a,
        input_b,
        input_core_dims=[["vector"], ["vector"]],
        output_core_dims=[["vector"]],
        dask="parallelized",
        output_dtypes=[input_a.dtype],
    )


def unit_vector(input_array: xr.DataArray):
    """
    Returns the unit vector in the direction of input_array
    """
    return input_array / vector_magnitude(input_array)


def scalar_projection(input_a: xr.DataArray, input_b: xr.DataArray):
    """
    Returns the magnitude of input_a which is parallel to input_b

    See https://en.wikipedia.org/wiki/Vector_projection for notation. scalar_projection is 'a_1'
    """
    return vector_dot(input_a, unit_vector(input_b))


def vector_projection(input_a: xr.DataArray, input_b: xr.DataArray):
    """
    Returns a vector which is the orthogonal projection of input_a onto a straight line parallel to input_b.
    It is parallel to input_b.

    See https://en.wikipedia.org/wiki/Vector_projection for notation. vector_rejection is 'vec(a)_1'
    """
    return scalar_projection(input_a, input_b) * unit_vector(input_b)


def vector_rejection(input_a: xr.DataArray, input_b: xr.DataArray):
    """
    Returns a vector which is the projection of input_a onto a straight line orthogonal to input_b.

    See https://en.wikipedia.org/wiki/Vector_projection for notation. vector_rejection is 'vec(a)_2'
    """
    return input_a - vector_projection(input_a, input_b)
