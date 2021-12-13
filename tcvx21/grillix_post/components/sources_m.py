"""
Routines to evaluate the density and temperature source strengths

Since the sources were used with fixed positions and widths, these parameters are given
default parameters matching the values used.
"""
import xarray as xr
import numpy as np


def smoothstep(test_point: float, step_centre: float, step_width: float, order=3):
    """
    Returns a hermite smoothstep of a specific order

    The step will be exactly 0 for test_point values less than step_centre - step_width / 2
    The step will be exactly 1 for test_point values greater than step_centre + step_width / 2

    For a step of order A, the (A-1)^th derivative will also be zero at the endpoints
    """

    xn = (np.atleast_1d(test_point) - step_centre) / step_width + 0.5

    # Ignore the runtime warning that comes from comparing NaN
    # NaN will return False in any comparison
    with np.errstate(invalid="ignore"):
        xn[xn <= 0] = 0
        xn[xn >= 1] = 1

    if order == 1:
        step_values = xn
    elif order == 2:
        step_values = -2.0 * xn ** 3 + 3.0 * xn ** 2
    elif order == 3:
        step_values = 6.0 * xn ** 5 - 15.0 * xn ** 4 + 10.0 * xn ** 3
    else:
        return NotImplemented

    if isinstance(test_point, xr.DataArray):
        return xr.DataArray(
            step_values,
            coords=test_point.coords,
            dims=test_point.dims,
            attrs=test_point.attrs,
        )
    else:
        return step_values


def density_source_function(
    rho, district, norm, source_strength, source_centre=0.915141, source_width=0.083797
):
    """Annular source for the density, mimicking neutral particle ionisation"""
    annular_source = xr.DataArray(
        source_strength
        * np.exp(-(((rho ** 2 - source_centre ** 2) / source_width) ** 2))
    )

    return xr.DataArray(annular_source.where(district == "CLOSED", 0.0)).assign_attrs(
        norm=norm.n0 / norm.tau_0
    )


def core_temperature_source_function(
    rho, district, norm, source_strength, source_centre=0.3, source_width=0.3
):
    core_source = source_strength * xr.DataArray(
        1 - smoothstep(rho, source_centre, source_width)
    )
    return xr.DataArray(core_source.where(district == "CLOSED", 0.0)).assign_attrs(
        norm=norm.Te0 / norm.tau_0
    )


def temperature_source_function(
    rho,
    district,
    norm,
    density: xr.DataArray,
    etemp: xr.DataArray,
    itemp: xr.DataArray,
    density_source: xr.DataArray,
    source_strength,
    source_centre=0.3,
    source_width=0.3,
):
    """
    Smooth-step core power injection, mimicking Ohmic power deposition
    The temperature source takes into account the power from the density source, to give a constant-power source
    """
    core_source = core_temperature_source_function(
        rho, district, norm, source_strength, source_centre, source_width
    )

    total_source = xr.DataArray(
        core_source / density - density_source * (etemp + itemp) / density
    ).assign_attrs(norm=norm.Te0 / norm.tau_0)
    core_source = xr.DataArray(core_source / density).assign_attrs(
        norm=norm.Te0 / norm.tau_0
    )
    annular_sink = xr.DataArray(
        -density_source * (etemp + itemp) / density
    ).assign_attrs(norm=norm.Te0 / norm.tau_0)

    return total_source, core_source, annular_sink


def _power_source_from_temperature_source(
    density_source: xr.DataArray,
    etemp_source: xr.DataArray,
    density: xr.DataArray,
    etemp: xr.DataArray,
    itemp: xr.DataArray,
    norm,
    kind: str = "total",
):
    """
    Returns the power source corresponding to a given density source and temperature source

    Can return the sum of the power contributions from the etemp and density sources, or just the etemp_source and
    density_source contributions, using the 'kind' parameter
    """
    if kind == "total":
        power_source = xr.DataArray(
            1.5 * (etemp_source * density + density_source * (etemp + itemp))
        )
    elif kind == "etemp_source":
        power_source = xr.DataArray(1.5 * (etemp_source * density))
    elif kind == "density_source":
        power_source = xr.DataArray(1.5 * (density_source * (etemp + itemp)))
    else:
        raise NotImplementedError(f"Power source kind {kind} not recognised")

    return power_source.assign_attrs(norm=norm.n0 * norm.Te0 / norm.tau_0)


def _integrated_power_source(
    grid,
    norm,
    density_source,
    etemp_source,
    mean_density,
    mean_etemp,
    mean_itemp,
    kind: str = "total",
):
    """
    Computes the integrated power source

    Can return the sum of the power contributions from the etemp and density sources, or just the etemp_source and
    density_source contributions, using the 'kind' parameter
    """
    from tcvx21.grillix_post.observables.integrations_m import (
        axisymmetric_cylindrical_integration,
    )

    power_source = _power_source_from_temperature_source(
        density_source,
        etemp_source,
        density=mean_density,
        etemp=mean_etemp,
        itemp=mean_itemp,
        norm=norm,
        kind=kind,
    )

    integrated_power_ = axisymmetric_cylindrical_integration(grid, norm, power_source)

    return integrated_power_.to("kW")


def integrated_sources(grid, equi, norm, params, snaps):
    """
    Compute the integrated particle and power sources

    Integrated power = core_power + density_source_power - edge_sink_power
                     = core_power (since the edge sink and density source powers cancel out)

    We check that this is the case, to within 1kW
    """
    from tcvx21.grillix_post.observables.integrations_m import (
        axisymmetric_cylindrical_integration,
    )

    rho = equi.normalised_flux_surface_label(grid.r_s, grid.z_s)

    density_source_rate = params["params_srcsnk"]["csrcn"]
    etemp_source_rate = params["params_srcsnk"]["csrcte"]

    density_source = density_source_function(
        rho, grid.districts, norm, density_source_rate
    )
    integrated_density_source = axisymmetric_cylindrical_integration(
        grid, norm, density_source
    )
    print(f"The integrated particle source rate is {integrated_density_source:4.3e}")

    mean_density = grid.shape(snaps.density.mean(dim=("phi", "tau")))
    mean_etemp = grid.shape(snaps.electron_temp.mean(dim=("phi", "tau")))
    mean_itemp = grid.shape(snaps.ion_temp.mean(dim=("phi", "tau")))

    etemp_source, etemp_core_source, etemp_edge_sink = temperature_source_function(
        rho=rho,
        district=grid.districts,
        norm=norm,
        density=mean_density,
        etemp=mean_etemp,
        itemp=mean_itemp,
        density_source=density_source,
        source_strength=etemp_source_rate,
    )

    # Compute the power injected into the core as the (positive) etemp source
    core_power = _integrated_power_source(
        grid,
        norm,
        density_source,
        etemp_core_source,
        mean_density,
        mean_etemp,
        mean_itemp,
        kind="etemp_source",
    )

    total_power = _integrated_power_source(
        grid,
        norm,
        density_source,
        etemp_source,
        mean_density,
        mean_etemp,
        mean_itemp,
        kind="total",
    )

    assert np.isclose(core_power.magnitude, total_power.magnitude, atol=1)

    # Density source power
    density_source_power = _integrated_power_source(
        grid,
        norm,
        density_source,
        etemp_source,
        mean_density,
        mean_etemp,
        mean_itemp,
        kind="density_source",
    )

    edge_sink_power = _integrated_power_source(
        grid,
        norm,
        density_source,
        etemp_edge_sink,
        mean_density,
        mean_etemp,
        mean_itemp,
        kind="etemp_source",
    )

    assert np.isclose((density_source_power + edge_sink_power).magnitude, 0, atol=1)

    print(
        f"The integrated power source is {total_power:4.3f}\n"
        f"The density source required {density_source_power:4.3f}, which was compensated by a temperature sink"
    )

    return {"density_source": integrated_density_source, "power_source": total_power}
