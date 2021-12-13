"""
Set up the "Pint" unit library

See unit_registry.txt for defined units (for reference only)
"""
import pint
import os
import warnings
import xarray as xr
import numpy as np

# Disable Pint's old fallback behavior (must come before importing Pint)
# Ignore pylint warning C0413: Import "import pint" should be placed at
# the top of the module
os.environ["PINT_ARRAY_PROTOCOL_FALLBACK"] = "0"

unit_registry = pint.UnitRegistry()
Quantity = unit_registry.Quantity
Dimensionless = Quantity(1, "")

# Silence NEP 18 warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

unit_registry.setup_matplotlib()

# Hide a warning that units are stripped when downcasting to numpy arrays
warnings.simplefilter("ignore", category=pint.errors.UnitStrippedWarning)


def convert_xarray_to_quantity(input_array: xr.DataArray) -> Quantity:
    """
    Converts an xarray with units attribute into a pint.Quantity with units (useful for unit checking)

    If the base array is already a Quantity, its units will be silently overwritten

    Note that all attributes except units will be lost.
    Not recommended except as a sanity check/in tests
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pint.errors.UnitStrippedWarning)
        return input_array.norm * np.asarray(input_array)
