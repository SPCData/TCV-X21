import numpy as np
from scipy.sparse import csr_matrix


def _ij_to_l_index(
    x_unstructured: np.ndarray, y_unstructured: np.ndarray, x_index: int, y_index: int
) -> int:
    """Converts (i, j) x/y indices to (l) grid indices"""

    grid_index = np.where(
        np.logical_and(
            x_unstructured[x_index] == x_unstructured,
            y_unstructured[y_index] == y_unstructured,
        )
    )[0]

    if len(grid_index) == 1:
        return grid_index[0]
    elif len(grid_index) == 0:
        return -1
    else:
        raise ValueError(
            "Multiple grid_indices cannot match a single (x_index, y_index)"
        )


def _find_nearest_neighbor(
    x_unstructured: np.ndarray,
    y_unstructured: np.ndarray,
    x_query: float,
    y_query: float,
) -> int:
    """Finds the nearest point to a given query point"""

    cartesian_distance = np.sqrt(
        (x_unstructured - x_query) ** 2 + (y_unstructured - y_query) ** 2
    )

    return np.argmin(cartesian_distance)


def _weightings_interpolate(
    x_unstructured: np.ndarray,
    y_unstructured: np.ndarray,
    x_query: float,
    y_query: float,
) -> tuple:
    """
    Returns the indices and weightings of neighbours for linear interpolation
    on the unstructured (x,y,z) tricolumn data
    """
    assert x_unstructured.ndim == 1
    assert y_unstructured.ndim == 1
    assert x_unstructured.size == y_unstructured.size

    if np.any(
        [
            x_query < x_unstructured.min(),
            x_query > x_unstructured.max(),
            y_query < y_unstructured.min(),
            y_query > y_unstructured.max(),
        ]
    ):
        return 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    diff_x = x_query - x_unstructured
    diff_y = y_query - y_unstructured

    ix0 = np.nanargmin(np.where(diff_x >= 0, diff_x, np.nan))
    ix1 = np.nanargmax(np.where(diff_x <= 0, diff_x, np.nan))

    iy0 = np.nanargmin(np.where(diff_y >= 0, diff_y, np.nan))
    iy1 = np.nanargmax(np.where(diff_y <= 0, diff_y, np.nan))

    i00 = _ij_to_l_index(x_unstructured, y_unstructured, ix0, iy0)
    i01 = _ij_to_l_index(x_unstructured, y_unstructured, ix0, iy1)
    i10 = _ij_to_l_index(x_unstructured, y_unstructured, ix1, iy0)
    i11 = _ij_to_l_index(x_unstructured, y_unstructured, ix1, iy1)

    # If any points not found
    if np.any(np.array([i00, i01, i10, i11]) < 0):
        nearest_neighbour = _find_nearest_neighbor(
            x_unstructured, y_unstructured, x_query, y_query
        )
        # Set the interpolation weighting to take a single point value
        return nearest_neighbour, 0, 0, 0, 1.0, 0.0, 0.0, 0.0

    # Find the distance from the cell edges to the query point
    dx0 = x_query - x_unstructured[ix0]
    dy0 = y_query - y_unstructured[iy0]
    dx1 = x_unstructured[ix1] - x_query
    dy1 = y_unstructured[iy1] - y_query

    # If edge-of-grid in x-direction, collapse to nearest neighbour in x
    if ix0 == ix1:
        wx0 = 1.0
        wx1 = 0.0
    else:
        wx0 = dx1 / (dx0 + dx1)
        wx1 = dx0 / (dx0 + dx1)

    # If edge-of-grid in y-direction, collapse to nearest neighbour in y
    if iy0 == iy1:
        wy0 = 1.0
        wy1 = 0.0
    else:
        wy0 = dy1 / (dy0 + dy1)
        wy1 = dy0 / (dy0 + dy1)

    return i00, i10, i01, i11, wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1


def make_matrix_interp(
    x_unstructured: np.ndarray,
    y_unstructured: np.ndarray,
    x_queries: np.ndarray,
    y_queries: np.ndarray,
) -> csr_matrix:
    """
    Makes a csr matrix which extracts the values at points defined
    by x_queries, y_queries.
    csr_matrix * unstructured_data = values at queries
    """
    assert x_unstructured.ndim == 1
    assert y_unstructured.ndim == 1
    assert x_unstructured.size == y_unstructured.size
    assert x_queries.ndim == 1
    assert y_queries.ndim == 1
    assert x_queries.size == y_queries.size

    nz = 0
    indi = []
    indj = []
    val = []

    # For each query, find the bilinear interpolation stencil
    for l in range(x_queries.size):
        indi.append(nz)

        i00, i10, i01, i11, w00, w10, w01, w11 = _weightings_interpolate(
            x_unstructured, y_unstructured, x_queries[l], y_queries[l]
        )

        for index, weight in zip([i00, i10, i01, i11], [w00, w10, w01, w11]):
            if weight > 0.0:
                nz += 1
                indj.append(index)
                val.append(weight)

    indi.append(nz)

    return csr_matrix((val, indj, indi), shape=(x_queries.size, x_unstructured.size))
