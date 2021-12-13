"""
Parametric spline interpolator

Useful when you need to interpolate a curve of values, like points along a flux surface or a chord
"""
import warnings
import numpy as np
from scipy.interpolate import splprep, splev


class ParametricSpline:
    """
    A wrapper class around slprep and splev from scipy.interpolate
    Uses cubic spline interpolation
    """

    def __init__(
        self,
        sample_points: list,
        t_points: np.array = None,
        smoothing: float = 0.0,
        periodic: bool = False,
        order: int = 3,
    ):
        """
        Calculates the knots and coefficients for a parametric cubic spline interpolator

        sample_points should be a list of 1D arrays, i.e. [x_points, y_points, ...]
        If t_points is given, then the arrays given in sample points should be parametrised by t

        If smoothing if > 0.0, then the input points will be smoothed. (Typically smoothing required << 1, recommend
        to check the fit)

        Order should be less than the number of sample points - 1
        """

        sample_length = None
        for sample_array in sample_points:
            if sample_length is None:
                # Use the first array to set the length
                sample_length = len(sample_array)
            else:
                assert len(sample_array) == sample_length

        assert sample_length > order, (
            f"Not enough sample points ({sample_length}) for an order ({order}) "
            "ParametricSpline"
        )

        if order < 3 and smoothing > 0.0:
            warnings.warn(
                UserWarning(
                    "Should not use smoothing for order < 3 in ParametricSpline"
                )
            )

        if t_points is None:
            tck, self.t_points = splprep(
                sample_points, s=smoothing, per=(1 if periodic else 0), k=order
            )
        else:
            tck, self.t_points = splprep(
                sample_points,
                u=t_points,
                s=smoothing,
                per=(1 if periodic else 0),
                k=order,
            )

        self.t_min, self.t_max = np.min(self.t_points), np.max(self.t_points)

        self.knots, self.coeffs, self.order = tck

    def __call__(self, t_evaluations: np.array):
        """
        Returns the spline evaluations at points given by t_evaluations
        N.b. if no t_points are provided to init, then the given sample points are assumed to be parametrised between 0 and 1
        """
        # Interpolation only!
        assert (
            t_evaluations.min() >= self.t_min and t_evaluations.max() <= self.t_max
        ), f"Requested points in the range\
            {t_evaluations.min()}, {t_evaluations.max()}, which is outside the interval {self.t_min}, {self.t_max}"

        return splev(t_evaluations, (self.knots, self.coeffs, self.order))
