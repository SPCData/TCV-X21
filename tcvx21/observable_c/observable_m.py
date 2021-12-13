"""
Simple data container for a observable
"""
from tcvx21 import Quantity
import numpy as np


class MissingDataError(Exception):
    """An error to indicate that the observable is missing data"""

    pass


class Observable:
    def __init__(self, data, diagnostic, observable, label, color, linestyle):
        """Simple container for individual observables"""

        try:
            self.name = data.observable_name
            self.label = label
            self.color = color
            self.linestyle = linestyle

            self.diagnostic, self.observable = diagnostic, observable
            self.dimensionality = data.dimensionality
            self.check_dimensionality()
            self.experimental_hierarchy = data.experimental_hierarchy

            self.simulation_hierarchy = getattr(data, "simulation_hierarchy", None)

            self._values = Quantity(data["value"][:], data["value"].units)
            try:
                self._errors = Quantity(data["error"][:], data["error"].units).to(
                    self._values.units
                )
            except IndexError:
                self._errors = Quantity(
                    np.zeros_like(self._values), data["value"].units
                ).to(self._values.units)

            self.mask = np.ones_like(self._values).astype(bool)

        except (AttributeError, IndexError):
            raise MissingDataError(
                f"Missing data for {diagnostic}:{observable}. Data available is {data}"
            )

    def check_dimensionality(self):
        raise NotImplementedError()

    @property
    def values(self) -> Quantity:
        """Returns the observable values, with a mask applied if applicable"""
        return self._values[self.mask]

    @property
    def errors(self) -> Quantity:
        """Returns the observable errors, with a mask applied if applicable"""
        return self._errors[self.mask]

    @property
    def units(self) -> str:
        """Returns the units of the values and errors, as a string"""
        return str(self._values.units)

    @property
    def is_empty(self):
        return False

    @property
    def has_errors(self):
        return bool(np.count_nonzero(self.errors))

    @property
    def compact_units(self) -> str:
        """Units with compact suffix"""
        if self.values.check("[length]^-3"):
            # Don't convert 10^19 m^-3 to ~10 1/µm^3
            return str(self.values.units)
        else:
            return str(np.max(np.abs(self.values)).to_compact().units)

    @property
    def npts(self):
        """Returns the number of unmasked observable points"""
        return self.values.size

    def nan_mask(self):
        """Returns a mask which will remove NaN values"""
        return np.logical_and(~np.isnan(self._values), ~np.isnan(self._errors))

    def check_attributes(self, other):
        self.mask = np.logical_and(self.mask, other.mask)
        assert self.color == other.color
        assert self.label == other.label
        assert self.dimensionality == other.dimensionality
        assert self.linestyle == other.linestyle
        if hasattr(self, "_positions_rsep"):
            assert np.allclose(
                self._positions_rsep, other._positions_rsep, equal_nan=True
            )
        if hasattr(self, "_positions_zx"):
            assert np.allclose(self._positions_zx, other._positions_zx, equal_nan=True)

    def fill_attributes(self, result):
        """Fills the attributes when copying to make a new object"""
        result.mask = self.mask
        result.color = self.color
        result.label = self.label
        result.dimensionality = self.dimensionality
        result.linestyle = self.linestyle

        if hasattr(self, "xmin") and hasattr(self, "xmax"):
            result.xmin, result.xmax, result.ymin, result.ymax = (
                self.xmin,
                self.xmax,
                None,
                None,
            )
        if hasattr(self, "_positions_rsep"):
            result._positions_rsep = self._positions_rsep
        if hasattr(self, "_positions_zx"):
            result._positions_zx = self._positions_zx

    def __add__(self, other):
        assert type(self) == type(other)
        result = object.__new__(self.__class__)
        result._values = self._values + other._values
        result._errors = np.sqrt(self._errors ** 2 + other._errors ** 2)
        self.fill_attributes(result)
        result.check_attributes(other)

        return result

    def __sub__(self, other):
        assert type(self) == type(other)
        result = object.__new__(self.__class__)
        result._values = self._values - other._values
        result._errors = np.sqrt(self._errors ** 2 + other._errors ** 2)
        self.fill_attributes(result)
        result.check_attributes(other)

        return result

    def __mul__(self, other):
        result = object.__new__(self.__class__)
        if isinstance(other, (float, Quantity)):
            # Scalar multiplication
            result._values = self._values * other
            result._errors = self._errors * other
            self.fill_attributes(result)

        else:
            assert type(self) == type(other)
            result._values = self._values * other._values
            result._errors = result._values * np.sqrt(
                (self._errors / self._values) ** 2
                + (other._errors / other._values) ** 2
            )
            self.fill_attributes(result)
            result.check_attributes(other)

        return result

    def __truediv__(self, other):
        assert type(self) == type(other)
        assert self._values.size == other._values.size

        result = object.__new__(self.__class__)
        result._values = self._values / other._values
        result._errors = result._values * np.sqrt(
            (self._errors / self._values) ** 2 + (other._errors / other._values) ** 2
        )
        self.fill_attributes(result)
        result.check_attributes(other)

        return result

    def trim_to_mask(self, mask):
        result = object.__new__(self.__class__)
        result._values = self._values[mask]
        result._errors = self._errors[mask]
        self.fill_attributes(result)
        result.mask = np.ones_like(result._values).astype(bool)

        if hasattr(self, "_positions_rsep"):
            result._positions_rsep = self._positions_rsep[mask]
        if hasattr(self, "_positions_zx"):
            result._positions_zx = self._positions_zx[mask]

        return result
