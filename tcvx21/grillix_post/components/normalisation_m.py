"""
Normalisation parameters, used for converting to SI units.

Based on the pint unit-handling library.

The typical usage is as follows;
1. Use read_normalisation_file to read a formatted fortran namelist, typically called 'physical_parameters.nml'
   and return a dictionary called 'normalisation'
2. Pass the 'normalisation' dictionary to dependent paramters
   i.e. rho_s0(**normalisation)
   which converts the dictionary into keyword arguments

Alternatively, one can manually generate the normalisation dictionary
"""
from pathlib import Path
import numpy as np
import xarray as xr
from tcvx21.units_m import Quantity as Q_, unit_registry as ureg
from .namelist_reader_m import read_fortran_namelist


class Normalisation:
    """
    dict-like class for converting from dimensionless arrays to SI units
    """

    @classmethod
    def initialise_from_normalisation_file(cls, normalisation_filepath: Path):
        """
        Reads from the normalisation namelist, and returns a dictionary of quantities
        """

        experiment_parameters = dict(
            read_fortran_namelist(normalisation_filepath)["physical_parameters"]
        )

        physical_parameters = {}

        for key in ["b0", "te0", "ti0", "n0", "r0", "mi", "z", "z_eff"]:
            assert key in experiment_parameters

        # Magnetic field normalisation, usually taken on axis, in Tesla
        physical_parameters["B0"] = Q_(experiment_parameters["b0"], "tesla")
        # Electron temperature normalisation, in electron-volts
        physical_parameters["Te0"] = Q_(experiment_parameters["te0"], "electron_volt")
        # Ion temperature normalisation, in electron-volts
        physical_parameters["Ti0"] = Q_(experiment_parameters["ti0"], "electron_volt")
        # Density normalisation, in particles-per-cubic-metres
        physical_parameters["n0"] = Q_(experiment_parameters["n0"], "metre**-3")
        # Major radius, in metres (n.b. this is also the scale length for parallel
        # quantities)
        physical_parameters["R0"] = Q_(experiment_parameters["r0"], "metre")
        # Ion mass, in amu
        physical_parameters["Mi"] = Q_(experiment_parameters["mi"], "proton_mass")
        # Ion charge
        physical_parameters["Z"] = experiment_parameters["z"]
        # Ion effective charge
        physical_parameters["Z_eff"] = experiment_parameters["z_eff"]

        return cls(physical_parameters)

    def __init__(self, normalisation_dictionary: dict):
        """
        Each element of the input normalisation dictionary is copied as an attribute of the object.

        i.e. If you define
        norm = Normalisation({'a':1, 'b':2})
        then
        norm.a == 1 and norm.b == 2
        """
        self.B0 = np.nan
        self.Te0 = np.nan
        self.Ti0 = np.nan
        self.n0 = np.nan
        self.R0 = np.nan
        self.Mi = np.nan
        self.Z = np.nan
        self.Z_eff = np.nan
        self.__dict__ = normalisation_dictionary

    def __repr__(self):
        """
        Returns a string representation for each element of the namelist
        """
        string = f"\t{'Value':<30}\t{'Magnitude':<10}\t{'Units':<20}\n"

        for parameter_name in sorted(dir(self), key=str.casefold):
            parameter = getattr(self, parameter_name, None)
            if isinstance(parameter, Q_):
                magnitude = parameter.magnitude
                units = str(parameter.units)

                string += f"\t{parameter_name:<30}\t{magnitude:<10.4e}\t{units:<30}\n"

        return string

    def __getitem__(self, key):
        """
        Allows normalisation to be indexed like a dictionary
        """
        return getattr(self, key)

    def convert_norm(self, input_array: xr.DataArray):
        """
        Converts the 'norm' attribute of a xarray from a string to a Quantity
        """
        assert isinstance(
            input_array, xr.DataArray
        ), "convert_norm requires an xr.DataArray input"
        assert (
            "norm" in input_array.attrs
        ), "convert_norm requires that the DataArray has an attribute 'norm'"

        if isinstance(input_array.norm, str):
            input_array.attrs["norm"] = getattr(self, input_array.norm)

        elif not isinstance(input_array.norm, Q_):
            raise NotImplementedError(
                f"Error: norm must be string or Quantity, but type was {type(input_array.norm)}"
            )

        return input_array

    def as_dict(self):
        return_dict = {}
        for parameter_name in sorted(dir(self), key=str.casefold):
            parameter = getattr(self, parameter_name, None)
            if isinstance(parameter, Q_):
                return_dict[parameter_name] = parameter

        return return_dict

    elementary_charge = Q_(1, "elementary_charge")
    electron_mass = Q_(1, "electron_mass")
    proton_mass = Q_(1, "proton_mass")
    atomic_mass_units = Q_(1, "amu")
    speed_of_light = Q_(1, "speed_of_light")
    vacuum_permeability = Q_(1, "mu0")
    vacuum_permittivity = Q_(1, "vacuum_permittivity")

    @property
    def rho_s0(self):
        """Ion Larmor Radius [m] (n.b. this is also the scale length for perpendicular quantities)"""
        return (
            np.sqrt(self.Ti0 * self.Mi) / (self.elementary_charge * self.B0)
        ).to_base_units()

    @property
    def c_s0(self):
        """Sound speed [m/s]"""
        return (np.sqrt(self.Te0 / self.Mi)).to_base_units()

    @property
    def v_A0(self):
        """Alfven speed [m/s]"""
        return (
            self.B0 / np.sqrt(self.vacuum_permeability * self.n0 * self.Mi)
        ).to_base_units()

    @property
    def tau_0(self):
        """Time normalisation [s]"""
        return (self.R0 / self.c_s0).to("s")

    @staticmethod
    @ureg.wraps(ret="", args=["cm**-3", "eV"], strict=True)
    def Coulomb_logarithm_ee(n, Te):
        """
        Coulomb logarithm (unitless) for thermal electron-electron collisions

        You can pass n and Te in any units, and these will be converted to
        cgs+eV such that the following formula applies
        """
        if Te > 10:
            return 24.0 - np.log(np.sqrt(n) / Te)
        else:
            return 23.0 - np.log(np.sqrt(n) / Te ** (1.5))

    @property
    def lnLambda0(self):
        """
        Thermal electron-electron Coulomb logarithm for reference parameters.

        Note that we usually the Coulomb logarithm to be constant to ensure that the
        collision operator is bilinear
        """
        return self.Coulomb_logarithm_ee(self.n0, self.Te0)

    def tau_ee(self, n, Te, Z):
        """
        Electron-electron collision time

        This is the mean time required for the direction of motion of an individual electron
        to be deflected by approximately 90 degrees due to collisions with electrons

        Using 3.182 from Fitzpatrick (in SI units), but taking the parametric dependency
        from equation 2.5e from Braginskii (n = n_e = Z n_i, so n_i = n/Z.
        Therefore Z**2 * n_i = Z n) -- since Braginskii is c.g.s. and conversion to s.i. is a pain
        """
        lnLambda = self.Coulomb_logarithm_ee(n, Te)

        return (
            (
                6.0
                * np.sqrt(2.0)
                * np.pi ** 1.5
                * self.vacuum_permittivity ** 2
                * np.sqrt(self.electron_mass)
                * Te ** 1.5
            )
            / (lnLambda * self.elementary_charge ** 4 * Z * n)
        ).to("s")

    def tau_ii(self, n, Te, Ti, Mi, Z):
        """
        Ion-ion collision time

        This is the mean time required for the direction of motion of an individual ion
        to be deflected by approximately 90 degrees due to collisions with ions

        Using 3.184 from Fitzpatrick (in SI units), but taking the parametric dependency from
        equation 2.5i from Braginskii (n = n_e = Z n_i, so n_i = n/Z.
        Therefore Z**4 * n_i = Z**3 n)
        """
        lnLambda = self.Coulomb_logarithm_ee(n, Te)

        return (
            (
                12.0
                * np.pi ** 1.5
                * self.vacuum_permittivity ** 2
                * np.sqrt(Mi)
                * Ti ** 1.5
            )
            / (lnLambda * self.elementary_charge ** 4 * Z ** 3 * n)
        ).to("s")

    @property
    def tau_e(self):
        """
        Reference electron-electron collision time, calculated with the effective charge Z_eff
        """
        return self.tau_ee(self.n0, self.Te0, self.Z_eff)

    @property
    def tau_i(self):
        """
        Reference ion-ion collision time
        """
        return self.tau_ii(self.n0, self.Te0, self.Ti0, self.Mi, self.Z)

    @property
    def delta(self):
        """ "
        Ratio of major radius to ion larmor radius.
        Conversion from perpendicular length scale to parallel length scale (>>1)
        """
        return self.R0 / self.rho_s0

    @property
    def rhostar(self):
        """
        Ratio of ion larmor radius to major radius (inverse delta)
        Conversion from parallel length scale to perpendicular length scale (<<1)
        """
        return self.rho_s0 / self.R0

    @property
    def beta_0(self):
        """Electron dynamical beta"""
        return self.c_s0 ** 2 / (self.v_A0 ** 2)

    @property
    def mu(self):
        """Electron to ion mass ratio"""
        return (self.electron_mass / self.Mi).to("")

    @property
    def zeta(self):
        """Ti0/Te0: Ion to electron temperature ratio"""
        return self.Ti0 / self.Te0

    @property
    def tau_e_norm(self):
        """Normalised electron collision time"""
        return self.tau_e * self.c_s0 / self.R0

    @property
    def tau_i_norm(self):
        """Normalised ion collision time"""
        return self.tau_i * self.c_s0 / self.R0

    @property
    def nu_e0(self):
        """Normalised electron collision frequency"""
        return self.tau_e_norm ** -1

    @property
    def nu_i0(self):
        """Normalised ion collision frequency"""
        return self.tau_i_norm ** -1

    @property
    def chipar_e(self):
        """Parallel electron heat conductivity"""
        return 3.16 * self.tau_e_norm * self.mu ** -1

    @property
    def chipar_i(self):
        """Parallel ion heat conductivity"""
        return 3.90 * self.zeta * self.tau_i_norm

    @property
    def etapar(self):
        """Parallel resistivity"""
        return 0.51 * self.nu_e0 * self.mu

    @property
    def ion_viscosity(self):
        """Ion viscosity coefficient"""
        return 0.96 * self.tau_i_norm
