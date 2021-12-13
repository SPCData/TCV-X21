"""
Implementation of a contour finding routine for 2D data
"""
import numpy as np
from skimage import measure


def find_contours(x_vals, y_vals, array, level: float) -> list:
    """
    Uses skimage.measure.find_contours to find the contours of an array at a given level

    skimage uses pixel units, so we need to convert back to real units
    """
    x_vals, y_vals, array = np.asarray(x_vals), np.asarray(y_vals), np.asarray(array)

    assert x_vals.size == array.shape[-1]
    assert y_vals.size == array.shape[-2]

    x_spacing, y_spacing = np.mean(np.diff(x_vals)), np.mean(np.diff(y_vals))
    assert np.allclose(np.diff(x_vals), x_spacing) and np.allclose(
        np.diff(y_vals), y_spacing
    ), "Error: basis vectors are not equally spaced."

    contours = measure.find_contours(array, level)
    # Contours is a list of numpy arrays, where the numpy arrays have the
    # shape (n, 2)

    for i, contour in enumerate(contours):
        # For each contour level found, switch the x and y elements, and then
        # convert to grid units

        x_contour, y_contour = contour[:, 1], contour[:, 0]
        x_contour, y_contour = (
            x_contour * x_spacing + x_vals.min(),
            y_contour * y_spacing + y_vals.min(),
        )
        contours[i] = np.column_stack((x_contour, y_contour))

    return contours
