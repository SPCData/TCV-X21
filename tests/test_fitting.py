import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tcvx21
from tcvx21 import Quantity
from tcvx21.analysis import (
    fit_eich_profile,
    fit_exponential_decay,
    make_decay_rate_table,
    make_eich_fit_comparison,
)
from tcvx21.analysis.fit_eich_profile_m import eich_profile
from tcvx21.analysis.fit_exp_decay_m import exponential_decay
from tcvx21.plotting import savefig


def compare_quantities(a, b, rtol=1e-3, atol=1e-06, equal_nan=True):
    b = b.to(a.units).magnitude
    a = a.magnitude
    print(f"{a:e}, {b:e}, {rtol:e}, {atol:e}, {a-b:e}, {(a-b)/a:e}")
    assert np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def test_fit_eich_profile():

    positions = Quantity(np.linspace(-2, 4), "cm")

    heat_flux = eich_profile(
        position=positions,
        peak_heat_flux=Quantity(5, "MW/m^2"),
        lambda_q=Quantity(1.2, "mm"),
        spreading_factor=Quantity(5.2, "mm"),
        background_heat_flux=Quantity(543, "kW/m^2"),
        shift=Quantity(1.2, "mm"),
    )

    (
        peak_heat_flux,
        lambda_q,
        spreading_factor,
        background_heat_flux,
        shift,
    ) = fit_eich_profile(position=positions, heat_flux=heat_flux)

    # See if the curve fitting can recover the profile if an exact solution is available
    # Each fitted parameter includes both the mean value and uncertainty of the fit
    compare_quantities(peak_heat_flux[0], Quantity(5, "MW/m^2"))
    compare_quantities(lambda_q[0], Quantity(1.2, "mm"))
    compare_quantities(spreading_factor[0], Quantity(5.2, "mm"))
    compare_quantities(background_heat_flux[0], Quantity(543, "kW/m^2"))
    compare_quantities(shift[0], Quantity(1.2, "mm"))

    np.all(
        np.array(
            [
                peak_heat_flux[1].magnitude,
                lambda_q[1].magnitude,
                spreading_factor[1].magnitude,
                background_heat_flux[1].magnitude,
                shift[1].magnitude,
            ]
        )
        < 1e-3
    )

    noisy_heat_flux = heat_flux * (
        1.0 + 5e-2 * np.random.default_rng(seed=1234).normal(size=positions.size)
    )
    (
        peak_heat_flux,
        lambda_q,
        spreading_factor,
        background_heat_flux,
        shift,
    ) = fit_eich_profile(position=positions, heat_flux=noisy_heat_flux, plot=True)
    savefig(plt.gcf(), output_path=tcvx21.test_figures_dir / "eich_fit.png")

    # See if the curve fitting can roughly recover the profile if noisy signal is provided
    # We expect that 95% of the data should be in 2*sigma, so in 95% of cases atol=2*sigma should pass

    compare_quantities(
        peak_heat_flux[0], Quantity(5, "MW/m^2"), atol=2 * peak_heat_flux[1].magnitude
    )
    compare_quantities(lambda_q[0], Quantity(1.2, "mm"), atol=2 * lambda_q[1].magnitude)
    compare_quantities(
        spreading_factor[0], Quantity(5.2, "mm"), atol=2 * spreading_factor[1].magnitude
    )
    compare_quantities(
        background_heat_flux[0],
        Quantity(543, "kW/m^2"),
        atol=2 * background_heat_flux[1].magnitude,
    )
    compare_quantities(shift[0], Quantity(1.2, "mm"), atol=2 * shift[1].magnitude)


def test_fit_exponential():

    positions = Quantity(np.linspace(-2, 4), "cm")

    values = Quantity(
        exponential_decay(
            positions,
            amplitude=Quantity(5.1e13, "m^-3"),
            decay_rate=Quantity(3.4, "millimeter"),
        )
    )

    A, l = fit_exponential_decay(position=positions, values=values, fit_logarithm=True)
    compare_quantities(A[0], Quantity(5.1e13, "m^-3"))
    compare_quantities(l[0], Quantity(3.4, "mm"))
    assert np.all(np.array([(A[1] / A[0]).magnitude, (l[1] / l[0]).magnitude]) < 1e-3)

    A, l = fit_exponential_decay(position=positions, values=values, fit_logarithm=False)
    compare_quantities(A[0], Quantity(5.1e13, "m^-3"))
    compare_quantities(l[0], Quantity(3.4, "mm"))
    assert np.all(np.array([(A[1] / A[0]).magnitude, (l[1] / l[0]).magnitude]) < 1e-3)

    noisy_values = values * (
        1.0 + np.random.default_rng(seed=1234).normal(size=positions.size)
    )
    A, l = fit_exponential_decay(
        position=positions,
        values=noisy_values,
        errors=noisy_values - values,
        plot="test_fit",
        fit_range=(0, 2),
    )
    savefig(plt.gcf(), output_path=tcvx21.test_figures_dir / "exp_fit.png")

    compare_quantities(A[0], Quantity(5.1e13, "m^-3"), rtol=0.25)
    compare_quantities(A[0], Quantity(5.1e13, "m^-3"), atol=2 * A[1].magnitude)

    compare_quantities(l[0], Quantity(3.4, "mm"), rtol=0.25)
    compare_quantities(l[0], Quantity(3.4, "mm"), atol=2 * l[1].magnitude)

    # Check that fitting with too-few points returns None
    A, l = fit_exponential_decay(
        position=positions, values=noisy_values, plot="test_fit", fit_range=(0, 0.2)
    )
    assert A[0] is None and l[0] is None


def test_make_fit_tables(comparison_data, tmpdir):

    make_decay_rate_table(
        **comparison_data,
        diagnostics=("TS",),
        observables=("density",),
        labels=("$n_{TS}$",),
        plot=True,
        output_path=Path(tmpdir) / "decay_rates.tex",
    )
    savefig(plt.gcf(), output_path=tcvx21.test_figures_dir / "exp_fit_comparison.png")
    plt.close("all")

    make_eich_fit_comparison(
        **comparison_data,
        fig_output_path=tcvx21.test_figures_dir / "eich_fit_comparison.png",
        table_output_path=Path(tmpdir) / "eich_fit.tex",
    )
