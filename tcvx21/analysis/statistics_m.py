import xarray as xr
from numpy.random import default_rng
from tcvx21.units_m import Dimensionless
from typing import Union
from scipy import stats
import numpy as np
import sys


def strip_moment(observable_key: str):
    """Convert a observable name to a base observable and a statistical moment"""

    if observable_key.endswith("_std"):
        moment = "std"
        key = observable_key.rstrip("std").rstrip("_")
    elif observable_key.endswith("_skew"):
        moment = "skew"
        key = observable_key.rstrip("skew").rstrip("_")
    elif observable_key.endswith("_kurtosis"):
        moment = "kurt"
        key = observable_key.rstrip("kurtosis").rstrip("_")
    else:
        moment = "mean"
        key = observable_key

    return key, moment


def compute_statistical_moment(values, moment: str = "mean", dimension=("tau", "phi")):
    """Compute the statistical moment"""
    if moment == "mean":
        return values.mean(dim=dimension)
    elif moment == "std":
        return values.std(dim=dimension)
    elif moment == "skew":
        return skew(values, dim=dimension)
    elif moment == "kurt":
        return kurtosis(values, dim=dimension, fisher=False)
    elif moment == "excess_kurt":
        return kurtosis(values, dim=dimension, fisher=True)
    else:
        raise NotImplementedError(f"No implementation for {moment}")


def compute_statistical_moment_with_bootstrap(
    values, moment: str = "mean", n_tests: int = 100, ci=0.95, random_seed=None
):
    """
    Return the statistical moment and an estimate of its uncertainty due to finite
    sample size effects
    """

    assert moment in ["mean", "std", "skew", "kurt", "excess_kurt"]

    if "pytest" in sys.modules:
        # Use a constant seed if using pytest, otherwise tests won't be reproducible
        rng = default_rng(seed=12345)
    else:
        rng = default_rng(seed=random_seed)

    stacked_values = values.stack(samples=("tau", "phi"))

    n_points = stacked_values.sizes["points"]
    n_samples = np.round(stacked_values.sizes["samples"]).astype(int)

    random_sample = rng.integers(
        low=0, high=stacked_values.sizes["samples"], size=(n_points, n_tests, n_samples)
    )

    random_sample = xr.DataArray(random_sample, dims=("points", "tests", "samples"))

    sampled_values = stacked_values.isel(samples=random_sample)

    test_moment = compute_statistical_moment(
        sampled_values, moment=moment, dimension="samples"
    )

    # What fraction of points should we exclude as outside our confidence interval (take factor of two since
    # we calculate a two-sided confidence interval).

    fraction_excluded = (1.0 - ci) / 2.0

    low, centre, high = test_moment.quantile(
        q=[fraction_excluded, 0.5, 1 - fraction_excluded], dim="tests"
    )

    return centre, np.abs(high - low)


def skew(
    input_array: xr.DataArray,
    dim: Union[str, tuple] = ("tau", "phi"),
    bias: bool = True,
    nan_policy: str = "propagate",
) -> xr.DataArray:
    """
    Calculates the sample skewness of a given xarray (Fisher-Pearson coefficient of skewness)

    For normally distributed data, the skewness should be about zero. For unimodal continuous distributions,
    a skewness value greater than zero means that there is more weight in the right tail of the distribution.

    Unlike the standard scipy.stats version, this function can s

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html for kwargs

    Note that the skew is dimensionless
    """

    assert isinstance(dim, str) or isinstance(
        dim, tuple
    ), f"dim must be either str or tuple, but was {dim} of type {type(dim)}"
    if isinstance(dim, tuple) and len(dim) > 1:
        # Can only supply a single dimension to reduce with scipy.stats. To circumvent this, we
        # 'stack' the multidimensions into a single dimension
        input_array = input_array.stack(stacked_dim=dim)
        dim = "stacked_dim"

    skew_array = input_array.reduce(
        func=stats.skew, dim=dim, keep_attrs=True, bias=bias, nan_policy=nan_policy
    )

    skew_array.attrs["norm"] = Dimensionless
    skew_array.attrs["units"] = ""
    return skew_array


def kurtosis(
    input_array: xr.DataArray,
    dim: Union[str, tuple] = ("tau", "phi"),
    fisher: bool = False,
    bias: bool = True,
    nan_policy: str = "propagate",
):
    """
    Calculates the sample kurtosis of a given xarray

    Kurtosis is the fourth central moment divided by the square of the variance.
    In Pearson's definition (fisher = False), this value is returned directly
    In Fisher’s definition (fisher = True), 3.0 is subtracted from the result to give 0.0 for a normal distribution.
    (Fisher's definition is typically termed the "excess kurtosis")

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html

    Note that the kurtosis is dimensionless
    """

    assert isinstance(dim, str) or isinstance(
        dim, tuple
    ), f"dim must be either str or tuple, but was {dim} of type {type(dim)}"
    if isinstance(dim, tuple) and len(dim) > 1:
        # Can only supply a single dimension to reduce with scipy.stats. To circumvent this, we
        # 'stack' the multidimensions into a single dimension
        input_array = input_array.stack(stacked_dim=dim)
        dim = "stacked_dim"

    kurt_array = input_array.reduce(
        func=stats.kurtosis,
        dim=dim,
        keep_attrs=True,
        fisher=fisher,
        bias=bias,
        nan_policy=nan_policy,
    )

    kurt_array.attrs["norm"] = Dimensionless
    kurt_array.attrs["units"] = ""
    return kurt_array
