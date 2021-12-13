"""
Routines to fit exponential decay functions to observables
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tcvx21 import Quantity


def exponential_decay(x, amplitude, decay_rate):
    """A curve given simple exponential decay"""
    return amplitude * np.exp(-x / decay_rate)


def linear_fit(x, intercept, gradient):
    return gradient * x + intercept


# K, A_log = np.polyfit(position, np.log(values), 1)
#     A = np.exp(A_log)
#     return A, K


def fit_exponential_decay(
    position,
    values,
    errors=None,
    fit_range=(-np.inf, np.inf),
    plot="",
    fit_logarithm=False,
):
    """
    Fit an exponential decay rate to some observable
    If you supply plot as a non-empty string, then this will be used to label a fit of the plot
    fit_range should be given as (min, max), in cm

    If supplying errors, use these in an absolute sense (i.e. don't rescale the errors to calculate a relative weighting
    of the points -- use the errors as directly setting the uncertainty of the fit)
    """

    position = position.to("cm").magnitude

    if errors is None:
        errors = np.ones_like(position)
        has_errors = False
    elif fit_logarithm:
        raise NotImplementedError("No implementation for fit_logarithm with errors")
    else:
        errors = errors.to(values.units).magnitude
        has_errors = True

    mask = np.logical_and.reduce(
        (
            position > fit_range[0],
            position < fit_range[1],
            ~np.isnan(values),
            ~np.isnan(errors),
        )
    )

    units = values.units
    values = values.magnitude[mask]
    position = position[mask]
    errors = errors[mask]

    if len(position) < 3:
        return (None, None), (None, None)

    if fit_logarithm:
        popt, pcov = curve_fit(linear_fit, xdata=position, ydata=np.log(values))
        perr = np.sqrt(np.diag(pcov))

        amplitude = np.exp(popt[0])
        amplitude_error = amplitude * perr[0]
        decay_rate = -1 / popt[1]
        decay_rate_error = np.abs(decay_rate * (perr[1] / popt[1]))

    else:
        popt, pcov = curve_fit(
            exponential_decay,
            xdata=position,
            ydata=values,
            sigma=errors,
            absolute_sigma=has_errors,
            p0=([np.max(values), 1.0]),
        )
        perr = np.sqrt(np.diag(pcov))
        amplitude, decay_rate = popt
        amplitude_error, decay_rate_error = perr

    if plot:
        plt.plot(position, exponential_decay(position, amplitude, decay_rate))
        if has_errors:
            plt.errorbar(position, values, errors, label=plot)
        else:
            plt.plot(position, values, label=plot)
        plt.gca().set_yscale("log", nonpositive="clip")

    return Quantity((amplitude, amplitude_error), units), Quantity(
        (decay_rate, decay_rate_error), "cm"
    )
