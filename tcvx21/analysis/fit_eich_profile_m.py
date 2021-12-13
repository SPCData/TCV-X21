"""
Routines to fit Eich-type heat flux profiles
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.optimize import curve_fit
from tcvx21 import Quantity
import warnings


def erfc(x):
    """
    Complimentary error function, with handling of Quantity objects
    Only Dimensionless quantities can be handled
    erfc(x) = 1 - erf(x)
    """
    if isinstance(x, Quantity):
        return scipy.special.erfc(x.to("").magnitude)
    else:
        return scipy.special.erfc(x)


def eich_profile(
    position, peak_heat_flux, lambda_q, spreading_factor, background_heat_flux, shift
):
    """
    Fits an Eich profile to heat flux data, using upstream-mapped
    distance to remove the explicit flux-expansion factor

    See Eich Nuclear Fusion 2013. Using the OMP-mapped distance as position
    means that you should set 'f_x' = 1

    You can pass the parameters to the profile either all as dimensional
    Quantity objects, or all as dimensionless raw arrays/floats

    position: OMP-mapped radial distance [mm]
    peak_heat_flux: peak heat flux [MW/m^2]
    lambda_q: heat flux decay width [mm]
    spreading_factor: spreading factor [mm]
    background_heat_flux: background heat-flux [MW/m^2]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (
            peak_heat_flux
            / 2.0
            * np.exp(
                (spreading_factor / (2.0 * lambda_q)) ** 2
                - (position - shift) / lambda_q
            )
            * erfc(
                spreading_factor / (2.0 * lambda_q)
                - (position - shift) / spreading_factor
            )
            + background_heat_flux
        )


def fit_eich_profile(position, heat_flux, errors=None, plot=False):
    """
    Uses non-linear least-squares to fit an Eich profile to heat flux data

    Should pass
    position: OMP-mapped radial distance [mm]
    heat_flux: total heat flux [MW/m^2]

    We use the Levenberg-Marquardt algorithm (default) for curve fitting

    Note that the algorithm takes into account the error if provided
    """

    if errors is None:
        errors = np.ones_like(position)
        has_errors = False
    else:
        errors = errors.to("MW/m^2").magnitude
        has_errors = True

    heat_flux = heat_flux.to("MW/m^2").magnitude

    mask = np.logical_and.reduce(
        (~np.isnan(heat_flux), ~np.isnan(errors), errors / heat_flux > 1e-2)
    )

    position = position.to("mm").magnitude[mask]
    heat_flux = heat_flux[mask]
    errors = errors[mask]

    popt, pcov = curve_fit(
        eich_profile,
        xdata=position,
        ydata=heat_flux,
        sigma=errors,
        absolute_sigma=has_errors,
        p0=(np.max(heat_flux), 5, 2, np.mean((heat_flux[0], heat_flux[-1])), 0.0),
        method="lm",
        xtol=1e-6,
    )

    peak_heat_flux, lambda_q, spreading_factor, background_heat_flux, shift = popt
    perr = np.sqrt(np.diag(pcov))
    (
        peak_heat_flux_err,
        lambda_q_err,
        spreading_err,
        background_heat_flux_err,
        shift_err,
    ) = perr

    if plot:
        plt.plot(
            Quantity(position, "mm"), Quantity(heat_flux, "MW/m^2"), label="signal"
        )
        if has_errors:
            plt.fill_between(
                Quantity(position, "mm"),
                Quantity(heat_flux + errors, "MW/m^2"),
                Quantity(heat_flux - errors, "MW/m^2"),
            )
        plt.plot(
            Quantity(position, "mm"),
            Quantity(eich_profile(position, *popt), "MW/m^2"),
            label="eich",
        )
        plt.legend()
        plt.title("Eich profile fitting")

    return (
        Quantity((peak_heat_flux, peak_heat_flux_err), "MW/m^2"),
        Quantity((lambda_q, lambda_q_err), "mm"),
        Quantity((spreading_factor, spreading_err), "mm"),
        Quantity((background_heat_flux, background_heat_flux_err), "MW/m^2"),
        Quantity((shift, shift_err), "mm"),
    )
