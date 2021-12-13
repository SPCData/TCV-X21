"""
Routines to calculate the heat flux profile and lambda_q
"""


def electron_parallel_heat_conduction(
    electron_temp, electron_temp_parallel_gradient, norm
):
    """
    Electron parallel heat conduction to the boundaries
    Equivalent to
    3.16 * density * electron_temp / (nu_ee * electron_mass) * parallel_grad(Te)
    with nu_ee the electron-electron collision rate
    """

    value = -norm.chipar_e * electron_temp ** 2.5 * electron_temp_parallel_gradient

    return value.assign_attrs(norm=(norm.n0 * norm.c_s0 * norm.Te0).to("MW/m^2"))


def ion_parallel_heat_conduction(ion_temp, ion_temp_parallel_gradient, norm):
    """
    Ion parallel heat conduction to the boundaries
    Equivalent to
    3.9 * density * ion_temp / (nu_ee * ion_mass) * parallel_grad(Ti)
    with nu_ee the ion-ion collision rate
    """

    value = -norm.chipar_i * ion_temp ** 2.5 * ion_temp_parallel_gradient

    return value.assign_attrs(norm=(norm.n0 * norm.c_s0 * norm.Ti0).to("MW/m^2"))


def electron_parallel_velocity(ion_velocity, current, density, norm):
    """
    Electron velocity parallel to the field
    """

    return ion_velocity - current / (norm.Z * density)


def electron_parallel_heat_convection(
    density, electron_temp, ion_velocity, current, norm
):
    """
    Electron parallel heat flux to the boundaries due to electron advection
    5/2 density * electron_velocity * electron_temp
    """

    vpar = electron_parallel_velocity(ion_velocity, current, density, norm)

    return (2.5 * density * electron_temp * vpar).assign_attrs(
        norm=(norm.n0 * norm.c_s0 * norm.Te0).to("MW/m^2")
    )


def ion_parallel_heat_convection(density, ion_temp, ion_velocity, norm):
    """
    Ion parallel heat flux to the boundaries due to ion advection
    5/2 density * ion_velocity * ion_temp
    """

    return (2.5 * density * ion_temp * ion_velocity).assign_attrs(
        norm=(norm.n0 * norm.c_s0 * norm.Ti0).to("MW/m^2")
    )


def exb_effective_parallel_heat_convection(
    density, temperature, effective_parallel_exb_velocity, norm
):
    """
    The effective parallel heat convection due to the ExB velocity (computed as the poloidal ExB component multiplied
    by the inverse magnetic field pitch)
    """
    return (2.5 * density * temperature * effective_parallel_exb_velocity).assign_attrs(
        norm=(norm.n0 * norm.c_s0 * temperature.norm).to("MW/m^2")
    )


def total_parallel_electron_heat_flux(
    density,
    electron_temp,
    electron_temp_parallel_gradient,
    ion_velocity,
    current,
    effective_parallel_exb_velocity,
    norm,
):
    """
    Sum of the conductive and convective components of the electron parallel heat flux to the boundaries
    """
    return (
        electron_parallel_heat_conduction(
            electron_temp, electron_temp_parallel_gradient, norm
        )
        + electron_parallel_heat_convection(
            density, electron_temp, ion_velocity, current, norm
        )
        + exb_effective_parallel_heat_convection(
            density, electron_temp, effective_parallel_exb_velocity, norm
        )
    )


def total_parallel_ion_heat_flux(
    density,
    ion_temp,
    ion_temp_parallel_gradient,
    ion_velocity,
    effective_parallel_exb_velocity,
    norm,
):
    """
    Sum of the conductive and convective components of the ion parallel heat flux to the boundaries
    """
    return (
        ion_parallel_heat_conduction(ion_temp, ion_temp_parallel_gradient, norm)
        + ion_parallel_heat_convection(density, ion_temp, ion_velocity, norm)
        + exb_effective_parallel_heat_convection(
            density, ion_temp, effective_parallel_exb_velocity, norm
        )
    )


def total_parallel_heat_flux(
    density,
    electron_temp,
    electron_temp_parallel_gradient,
    ion_temp,
    ion_temp_parallel_gradient,
    ion_velocity,
    current,
    effective_parallel_exb_velocity,
    norm,
):
    """
    Total heat flux -- sum of the electron and ion heat fluxes
    """
    electron_heat_flux = total_parallel_electron_heat_flux(
        density,
        electron_temp,
        electron_temp_parallel_gradient,
        ion_velocity,
        current,
        effective_parallel_exb_velocity,
        norm,
    )
    ion_heat_flux = total_parallel_ion_heat_flux(
        density,
        ion_temp,
        ion_temp_parallel_gradient,
        ion_velocity,
        effective_parallel_exb_velocity,
        norm,
    )

    return electron_heat_flux + ion_heat_flux
