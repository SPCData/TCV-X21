from tcvx21 import Quantity
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from .observable_m import Observable, MissingDataError
from matplotlib.colors import Normalize, LogNorm


class Observable2D(Observable):
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

        try:
            super().__init__(data, diagnostic, observable, label, color, linestyle)
            self._positions_rsep = Quantity(
                data["Rsep_omp"][:], data["Rsep_omp"].units
            ).to("cm")
            self._positions_zx = Quantity(data["Zx"][:], data["Zx"].units).to("m")

            try:
                # If the (R, Z) coordinates for the reference equilibrium are available, also load them
                self._positions_r = Quantity(data["R"][:], data["R"].units).to("m")
                self._positions_z = Quantity(data["Z"][:], data["Z"].units).to("m")
            except IndexError:
                pass

            self.set_mask()

        except AttributeError:
            raise MissingDataError

    def check_dimensionality(self):
        assert self.dimensionality == 2

    @property
    def positions_rsep(self) -> Quantity:
        """
        Returns the radial observable positions (using the flux-surface-label R^u - R^u_omp)
        with a mask applied if applicable
        """
        return self._positions_rsep[self.mask]

    @property
    def positions_zx(self) -> Quantity:
        """
        Returns the vertical observable positions (using the x-point displacement Z - Z_x)
        with a mask applied if applicable
        """
        return self._positions_zx[self.mask]

    def set_mask(
        self,
        rsep_min=Quantity(-np.inf, "m"),
        rsep_max=Quantity(np.inf, "m"),
        zx_min=Quantity(-np.inf, "m"),
        zx_max=Quantity(np.inf, "m"),
    ):
        """Constructs an array mask for returning values in a diagnostic of interest, and removing NaN values"""

        position_mask = np.logical_and.reduce(
            (
                self._positions_rsep > rsep_min,
                self._positions_rsep < rsep_max,
                self._positions_zx > zx_min,
                self._positions_zx < zx_max,
            )
        )

        nan_mask = self.nan_mask()

        self.mask = np.logical_and.reduce((position_mask, nan_mask))

    def plot(
        self,
        ax: plt.Axes = None,
        plot_type: str = "values",
        units: str = None,
        cbar_lim=None,
        log_cbar: bool = False,
        **kwargs,
    ):
        """
        Plots the observable
        """
        if ax is None:
            ax = plt.gca()

        if plot_type == "values":
            image = self.plot_values(
                ax, units=units, cbar_lim=cbar_lim, log_cbar=log_cbar, **kwargs
            )
        elif plot_type == "errors":
            image = self.plot_values(
                ax,
                show_errors=True,
                units=units,
                cbar_lim=cbar_lim,
                log_cbar=log_cbar,
                **kwargs,
            )
        elif plot_type == "sample_points":
            image = self.plot_sample_points(ax)
        else:
            raise NotImplementedError(f"No implementation for plot_type = {plot_type}")

        ax.set_xlabel("$R^u - R^u_{sep}$" + f" [{Quantity(1, ax.xaxis.units).units:~P}]")
        ax.set_ylabel("$Z - Z_X$" + f" [{Quantity(1, ax.yaxis.units).units:~P}]")

        return image

    def plot_sample_points(self, ax):
        """
        Plot the positions where the data is defined
        """
        return ax.scatter(
            self.positions_rsep, self.positions_zx, color=self.color, label=self.label
        )

    def _get_gridded_values(
        self,
        rsep_samples: int = 100,
        zx_samples: int = 150,
        errors: bool = False,
        units: str = None,
    ) -> [Quantity, Quantity, Quantity]:
        """Interpolates the values onto a regular 2D mesh"""
        rsep_basis = np.linspace(
            self.positions_rsep.min(), self.positions_rsep.max(), num=rsep_samples
        )
        zx_basis = np.linspace(
            self.positions_zx.min(), self.positions_zx.max(), num=zx_samples
        )

        rsep_mesh, zx_mesh = np.meshgrid(rsep_basis, zx_basis)

        gridded_values = griddata(
            (self.positions_rsep.magnitude, self.positions_zx.magnitude),
            self.values.magnitude if not errors else self.errors.magnitude,
            (rsep_mesh.magnitude, zx_mesh.magnitude),
            method="linear",
            rescale=True,
        )

        gridded_values = Quantity(gridded_values, self.units).to(
            self.compact_units if not units else units
        )

        return rsep_basis, zx_basis, gridded_values

    def plot_values(
        self,
        ax,
        show_errors: bool = False,
        units: str = None,
        cbar_lim=None,
        log_cbar: bool = False,
        n_contours=15,
        diverging: bool = False,
        robust: bool = True,
        **kwargs,
    ):
        """
        Plot the values or errors as a 2D filled grid
        """
        from tcvx21.plotting.labels_m import make_colorbar

        rsep_basis, zx_basis, gridded_values = self._get_gridded_values(
            errors=show_errors, units=units
        )

        if robust:
            # Use the 2% and 98% quantiles, rather than min and max, to remove outliers
            vmin, vmax = np.nanquantile(gridded_values, q=[0.02, 0.98]).magnitude
        else:
            vmin, vmax = np.nanquantile(gridded_values, q=[0.0, 1.0]).magnitude
        if cbar_lim is not None:
            vmin, vmax = cbar_lim.to(gridded_values.units).magnitude
        if diverging:
            abs_vmax = max(np.abs(vmin), np.abs(vmax))
            vmin, vmax = -abs_vmax, abs_vmax

        if not log_cbar:
            cnorm = Normalize(vmin, vmax)
            levels = np.linspace(vmin, vmax, num=n_contours, endpoint=True)
        elif vmin < 0.0 or diverging:
            raise NotImplementedError("Symmetric logarithmic plots not supported")
        else:
            cnorm = LogNorm(vmin, vmax)
            levels = np.logspace(
                np.log10(vmin), np.log10(vmax), num=n_contours, endpoint=True
            )

        image = ax.contourf(
            rsep_basis,
            zx_basis,
            gridded_values.magnitude,
            norm=cnorm,
            levels=levels,
            extend="both",
            **kwargs,
        )

        ax.xaxis.units, ax.yaxis.units = str(rsep_basis.units), str(zx_basis.units)
        ax.set_title(self.label)

        if cbar_lim is None:
            cbar = make_colorbar(
                ax, mappable=image, units=gridded_values.units, as_title=True
            )
            if log_cbar:
                cbar.ax.set_yticklabels("")

        return image

    def calculate_cbar_limits(
        self, others: list = None, robust: bool = True
    ) -> Quantity:
        """
        Calculates a constant colormap normalisation which captures the range of all values, including from other
        observables passed as others
        """
        if others is None:
            others = []

        values = self.values

        for other in others:
            if other.is_empty:
                continue

            values = np.append(values, other.values)

        if robust:
            return np.quantile(values, q=[0.02, 0.98])
        else:
            return np.quantile(values, q=[0.0, 1.0])

    def interpolate_onto_reference(self, reference, plot_comparison: bool = False):
        """
        Linearly interpolates a 2D array of simulation values onto the points where reference data is
        given. Extrapolation of simulation data is not allowed. NaNs will be returned for points
        outside the convex hull of points

        It is assumed that the simulation data is at a superset of the experimental points.

        If you would like to check the interpolation, you can set the plot_comparison flag to true
        """

        rs_ref, rs_test = (
            reference.positions_rsep.magnitude,
            self.positions_rsep.magnitude,
        )
        zx_ref, zx_test = reference.positions_zx.magnitude, self.positions_zx.magnitude

        interpolated_value = Quantity(
            griddata((rs_test, zx_test), self.values.magnitude, (rs_ref, zx_ref)),
            self.units,
        )
        interpolated_error = Quantity(
            griddata((rs_test, zx_test), self.errors.magnitude, (rs_ref, zx_ref)),
            self.units,
        )

        if plot_comparison:
            plt.plot(np.arange(reference.npts), reference.values, label="reference")
            plt.fill_between(
                np.arange(reference.npts),
                reference.values + reference.errors,
                reference.values - reference.errors,
            )
            plt.plot(np.arange(reference.npts), interpolated_value, label="test")
            plt.fill_between(
                np.arange(reference.npts),
                interpolated_value + interpolated_error,
                interpolated_value - interpolated_error,
            )
            plt.legend()

        result = object.__new__(self.__class__)
        self.fill_attributes(result)
        result._positions_rsep = rs_ref
        result._positions_zx = zx_ref
        result._values = interpolated_value
        result._errors = interpolated_error
        result.mask = ~np.logical_or(
            np.isnan(interpolated_value), np.isnan(interpolated_error)
        )

        return result
