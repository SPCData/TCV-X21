"""
Common experimental observables
"""
import xarray as xr


def ion_saturation_current(
    density: xr.DataArray, sound_speed: xr.DataArray, norm, wall_probe: bool
):
    """
    Ion saturation current for a Langmuir probe, usually referred to as 'Jsat' or 'Isat'

    If the probe is mounted on the wall, set wall_probe = True to remove the 1/2 factor required for
    an immersed probe
    """
    return xr.DataArray(
        density * sound_speed * (1.0 if wall_probe else 0.5)
    ).assign_attrs(
        norm=(norm.c_s0 * norm.elementary_charge * norm.n0).to("kiloampere*meter**-2"),
        name="Ion saturation current",
    )


def floating_potential(
    potential: xr.DataArray,
    electron_temp: xr.DataArray,
    norm,
    sheath_potential: float = 2.69,
):
    """
    Floating potential: the potential at which no current will flow into a probe
    """

    return xr.DataArray(potential - electron_temp * sheath_potential).assign_attrs(
        norm=potential.norm, name="Sheath potential"
    )
