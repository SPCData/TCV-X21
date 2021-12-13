"""
Parallel velocity and ExB velocity, as vector arrays
"""
import xarray as xr
import numpy as np
from tcvx21.grillix_post.components.vector_m import (
    toroidal_vector,
    vector_cross,
    vector_dot,
)
from tcvx21.grillix_post.observables.perpendicular_gradient_m import (
    perpendicular_gradient,
)


def parallel_ion_velocity_vector(grid, equi, velocity: xr.DataArray):
    """
    Converts the parallel ion velocity ion a vector array
    """

    return xr.DataArray(
        equi.parallel_unit_vector(grid.r_u, grid.z_u, grid=False) * velocity
    ).assign_attrs(norm=velocity.norm, name="Parallel velocity vector")


def sound_speed(
    electron_temp: xr.DataArray,
    ion_temp: xr.DataArray,
    norm,
    adiabatic_index: float = 1.0,
):
    """
    Ion sound speed for longitudinal waves.

    Use adiabatic_index = 0.0 to exclude ion_temperature from the calculation
    Normally use adiabatic_index = 1.0, but it is possible that 3.0 could also be valid since some kinetic modelling
    suggests ions have an additional degree of freedom
    """

    return xr.DataArray(
        np.sqrt(
            electron_temp + adiabatic_index * norm.Z * ion_temp / norm.zeta.magnitude
        )
    ).assign_attrs(norm=norm.c_s0, name="Ion sound speed")


def electric_field(grid, norm, potential: xr.DataArray):
    """
    Returns the electric field calculated from the scalar potential, as a vector array

    The potential gradient is normalised to the perpendicular length scale (i.e. units of phi0 / rho_s0)
    """
    shaped_potential = grid.shape(potential)
    assert shaped_potential.norm

    potential_gradient = perpendicular_gradient(grid, norm, shaped_potential)

    return grid.flatten(xr.DataArray(potential_gradient * -1.0)).assign_attrs(
        norm=potential_gradient.norm, name="Electric field"
    )


def exb_velocity(grid, equi, norm, potential: xr.DataArray):
    """
    Returns the E cross B velocity as a vector array.
    Assumes that the magnetic field can be approximated as purely toroidal.

    It is possible to calculate the exb_velocity with the full magnetic field, but the result is no longer within a
    poloidal plane.

    Units
    [vE] = [B]/[B]**2 * [grad_perp] * [phi]
         = B0 / B0**2 * 1/rho_s0 * Te0/e
         = Te0/(e * rho_s0 * B0)
    We can use rho_s0 = sqrt(Te0 * Mi)/(e * B0), so
    [vE] = sqrt(Te0 / Mi)
         = c_s0

    Need to multiply by a factor of delta to get to perpendicular velocity scale (which is what is used in GRILLIX).
    However, in order to easily compare to parallel velocities, we leave the drift velocity normalised to c_s0
    """

    E_field = electric_field(grid, norm, potential)

    B_toroidal = norm.convert_norm(
        equi.magnetic_field_toroidal(grid.r_u, grid.z_u, grid=False)
    )

    v_E = (
        (vector_cross(E_field, toroidal_vector(B_toroidal)) / B_toroidal ** 2)
        .assign_attrs(norm=E_field.norm / B_toroidal.norm, name="ExB velocity")
        .persist()
    )

    return v_E


def effective_parallel_exb_velocity(grid, equi, norm, potential: xr.DataArray):
    """
    The ExB velocity has components both across and along magnetic flux surfaces.

    For the along-flux-surface component, we are interested in how this compares to the parallel transport.

    We can consider this in one of two ways. We can compute the total_velocity, which is a 3D vector, and then
    use vector_dot to project this into some direction of interest.

    Another approach is to consider what parallel velocity would lead to the same poloidal transport as the ExB
    velocity. This is the effective parallel ExB velocity (it is NOT the parallel component of the ExB, which is small)

    We first compute the poloidal component of the ExB velocity, and then use the inverse magnetic field pitch to
    find the equivalent parallel vector (essentially, calculating the hypotenuse)

    Because this term is primarily useful for comparing to diagnostics, we return as a flattened array
    """

    v_E = exb_velocity(grid, equi, norm, potential)

    e_pol = equi.poloidal_unit_vector(grid.r_u, grid.z_u, grid=False)

    b_x = equi.magnetic_field_r(grid.r_u, grid.z_u, grid=False)
    b_y = equi.magnetic_field_z(grid.r_u, grid.z_u, grid=False)
    b_t = equi.magnetic_field_toroidal(grid.r_u, grid.z_u, grid=False)

    return vector_dot(v_E, e_pol) * np.sqrt(
        (b_x ** 2 + b_y ** 2 + b_t ** 2) / (b_x ** 2 + b_y ** 2)
    )


def total_velocity(grid, equi, norm, potential: xr.DataArray, velocity: xr.DataArray):
    """Returns the sum of the ExB and parallel velocity (it is assumed that the diamagnetic component is small"""

    v_E = exb_velocity(grid, equi, norm, potential)
    u_par = parallel_ion_velocity_vector(grid, equi, velocity)

    # The ExB velocity norm is equal to the sound speed. We can check this
    assert np.isclose(norm.c_s0, v_E.norm)
    assert np.isclose(norm.c_s0, u_par.norm)

    return (v_E + u_par).assign_attrs(name="Total velocity", norm=norm.c_s0)
