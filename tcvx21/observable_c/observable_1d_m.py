"""
A class for 1D observables like the density profile along a lineout
"""

from tcvx21 import Quantity
from tcvx21.units_m import pint
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .observable_m import Observable, MissingDataError


class Observable1D(Observable):
    def __init__(self, data, diagnostic, observable, label, color, linestyle):
        self.name = ""
        self.label = ""
        self.color = ""
        self.linestyle = ""
        self.diagnostic = ""
        self.observable = ""
        self.dimensionality = -1
        self.experimental_hierarchy = -1
        self.simulation_hierarchy = -1
        self._values = []
        self._errors = []
        self.mask = []

        super().__init__(data, diagnostic, observable, label, color, linestyle)
        self._positions_rsep = Quantity(data["Rsep_omp"][:], data["Rsep_omp"].units).to(
            "cm"
        )
        self.set_mask()
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None

    def check_dimensionality(self):
        assert self.dimensionality == 1

    @property
    def positions(self) -> Quantity:
        """
        Returns the observable positions (using the flux-surface-label R^u - R^u_omp
        with a mask applied if applicable
        """
        return self._positions_rsep[self.mask]

    def _position_mask(
        self, position_min=Quantity(-np.inf, "m"), position_max=Quantity(np.inf, "m")
    ):
        return np.logical_and(
            self._positions_rsep > position_min,
            self._positions_rsep < position_max,
        )

    def set_mask(
        self, position_min=Quantity(-np.inf, "m"), position_max=Quantity(np.inf, "m")
    ):
        """Constructs an array mask for returning values in a diagnostic of interest, and removing NaN values"""

        self.mask = np.logical_and.reduce(
            (self._position_mask(position_min, position_max), self.nan_mask())
        )

    @property
    def xlim(self):
        """Returns a tuple giving the bounds of the positions (after the mask is applied)"""
        return (
            Quantity(self.xmin, self.positions.units)
            if self.xmin is not None
            else np.min(self.positions),
            Quantity(self.xmax, self.positions.units)
            if self.xmax is not None
            else np.max(self.positions),
        )

    def set_plot_limits(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """Sets value limits (i.e. ylimits) for plotting"""
        if xmin is not None:
            self.xmin = xmin
        if xmax is not None:
            self.xmax = xmax
        if ymin is not None:
            self.ymin = ymin
        if ymax is not None:
            self.ymax = ymax

    def apply_plot_limits(self, ax):
        """Sets the plot limits if they are non-None"""
        if self.xmin is not None:
            ax.set_xlim(left=self.xmin)

        if self.xmax is not None:
            ax.set_xlim(right=self.xmax)

        if self.ymin is not None:
            ax.set_ylim(bottom=self.ymin)

        if self.ymax is not None:
            ax.set_ylim(top=self.ymax)

    def trim_mask(self, trim_to_x: tuple, trim_expansion=0.1):
        trim_to_x = list(trim_to_x)
        assert len(trim_to_x) == 2

        if not isinstance(trim_to_x[0], Quantity):
            trim_to_x[0] = Quantity(trim_to_x[0], "cm")
        if not isinstance(trim_to_x[1], Quantity):
            trim_to_x[1] = Quantity(trim_to_x[1], "cm")

        trim_range = trim_to_x[0] - trim_expansion * (
            trim_to_x[1] - trim_to_x[0]
        ), trim_to_x[1] + trim_expansion * (trim_to_x[1] - trim_to_x[0])
        mask = np.logical_and(self.mask, self._position_mask(*trim_range)).astype(bool)
        return mask

    def values_in_trim(self, trim_to_x: tuple, trim_expansion=0.1):
        mask = self.trim_mask(trim_to_x, trim_expansion)
        return self._values[mask]

    def ylim_in_trim(self, trim_to_x: tuple, trim_expansion=0.1):
        values = self.values_in_trim(trim_to_x, trim_expansion)
        return values.min(), values.max()

    def ylims_in_trim(
        self, others, trim_to_x: tuple, trim_expansion=0.1, range_expansion=0.1
    ):
        ymin, ymax = Quantity(np.zeros(len(others) + 1), self.units), Quantity(
            np.zeros(len(others) + 1), self.units
        )

        ymin[0], ymax[0] = self.ylim_in_trim(trim_to_x, trim_expansion)

        for i, case in enumerate(others):
            ymin[i + 1], ymax[i + 1] = case.ylim_in_trim(trim_to_x, trim_expansion)

        ymin, ymax = ymin.min(), ymax.max()

        return ymin - range_expansion * (ymax - ymin), ymax + range_expansion * (
            ymax - ymin
        )

    def plot(
        self,
        ax: plt.Axes = None,
        plot_type: str = "region",
        trim_to_x: tuple = tuple(),
        trim_expansion=0.1,
        **kwargs,
    ):
        """
        Plots the observable

        Can set "plot_type" to give either an errorbar plot or a shaded region around a mean line

        If trim_to_x is passed, it should be a tuple of length 2, giving the min and max positions for the
        values to plot.
        """
        if ax is None:
            ax = plt.gca()

        original_mask = np.copy(self.mask)

        if trim_to_x:
            self.mask = self.trim_mask(trim_to_x, trim_expansion)

        if plot_type == "errorbar":
            line = self.plot_errorbar(ax, **kwargs)
        elif plot_type == "region":
            line = self.plot_region(ax, **kwargs)
        else:
            raise NotImplementedError(f"No implementation for plot_type = {plot_type}")

        self.apply_plot_limits(ax)

        ax.set_xlabel("$R^u - R^u_{sep}$" + f" [{ax.xaxis.units:~P}]")

        self.mask = original_mask

        return line

    def plot_errorbar(
        self,
        ax: plt.Axes,
        color=None,
        label=None,
        linestyle=None,
        errorevery=5,
        **kwargs,
    ):
        """
        Makes an errorbar plot of the observable (not recommended)
        """
        return ax.errorbar(
            self.positions,
            self.values.to(self.compact_units),
            self.errors.to(self.compact_units),
            color=self.color if color is None else color,
            label=self.label if label is None else label,
            linestyle=self.linestyle if linestyle is None else linestyle,
            errorevery=errorevery,
            **kwargs,
        )[0]

    def plot_region(
        self, ax: plt.Axes, color=None, label=None, linestyle=None, **kwargs
    ):
        """
        Makes a plot of the observable, with the error represented by a shaded region around the line
        """
        (line,) = ax.plot(
            self.positions,
            self.values.to(self.compact_units),
            color=self.color if color is None else color,
            label=self.label if label is None else label,
            linestyle=self.linestyle if linestyle is None else linestyle,
            **kwargs,
        )

        ax.fill_between(
            self.positions,
            self.values + self.errors,
            self.values - self.errors,
            alpha=0.25,
            color=self.color if color is None else color,
        )
        return line

    @staticmethod
    def _interpolate_points(x_source, y_source, x_query):
        """Calls interp1d for quantities"""
        interpolator = interp1d(
            x=x_source.to(x_query.units).magnitude,
            y=y_source.magnitude,
            kind="cubic",
            bounds_error=True,
        )

        interpolated = interpolator(np.array(x_query.magnitude))
        return Quantity(interpolated, y_source.units)

    def points_overlap(self, reference_positions):
        """
        Returns a boolean array which can be used to mask reference values so that interpolate_onto_positions
        only interpolates (no extrapolation)
        """
        return np.logical_and(
            reference_positions > self.positions.min(),
            reference_positions < self.positions.max(),
        )

    def interpolate_onto_positions(self, reference_positions):
        """
        Interpolates a 1D array of simulation values onto the points where reference data is
        given. Extrapolation of simulation data is not allowed: instead, the reference data
        is cropped to the range of the simulation data.
        """

        interpolated_value = self._interpolate_points(
            self.positions, self.values, reference_positions
        )

        interpolated_error = self._interpolate_points(
            self.positions, self.errors, reference_positions
        )

        assert np.allclose(
            reference_positions.size, interpolated_value.size, interpolated_error.size
        )
        result = object.__new__(self.__class__)
        self.fill_attributes(result)

        result._positions_rsep = reference_positions
        result._values = interpolated_value
        result._errors = interpolated_error
        result.mask = np.ones_like(reference_positions).astype(bool)

        return result
