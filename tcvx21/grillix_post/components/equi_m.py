"""
Interface to the magnetic equilibrium
"""
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from tcvx21.units_m import Quantity, convert_xarray_to_quantity, Dimensionless
from tcvx21.analysis.contour_finding_m import find_contours
from .vector_m import poloidal_vector, cylindrical_vector


def array_coords(array, grid, r_norm, z_norm):
    """Sets the attributes of an array based on whether it returns with grid=True"""

    if grid:
        return array.rename({"dim_0": "Z", "dim_1": "R"}).assign_coords(
            R=r_norm, Z=z_norm
        )
    else:
        return array.rename({"dim_0": "points"})


class Equi:
    def __init__(
        self, equi_file: Path, penalisation_file: Path = None, flip_z: bool = False
    ):
        """
        Initialises an object which stores information about the magnetic geometry and boundaries
        """
        assert equi_file.exists()

        magnetic_geometry = xr.open_dataset(equi_file, group="Magnetic_geometry")
        psi_limits = xr.open_dataset(equi_file, group="Psi_limits")

        if penalisation_file is not None:
            assert penalisation_file.exists()
            penalisation_file = xr.open_dataset(penalisation_file)
            self.penalisation_available = True

            self.penalisation_characteristic = penalisation_file["pen_chi"].rename(
                {"dim_nl": "points"}
            )
            self.penalisation_direction = penalisation_file["pen_xi"].rename(
                {"dim_nl": "points"}
            )

        else:
            self.penalisation_available = False
            self.penalisation_characteristic, self.penalisation_direction = None, None

        self.magnetic_geometry = magnetic_geometry
        self.psi_limits = psi_limits

        self.poloidal_field_factor = +1.0
        self.flipped_z = False

        self.axis_r = xr.DataArray(magnetic_geometry.magnetic_axis_R).assign_attrs(
            norm=Quantity(1, magnetic_geometry.magnetic_axis_units)
        )
        self.axis_z = xr.DataArray(magnetic_geometry.magnetic_axis_Z).assign_attrs(
            norm=Quantity(1, magnetic_geometry.magnetic_axis_units)
        )

        self.x_point_r = xr.DataArray(
            np.atleast_1d(magnetic_geometry.x_point_R)
        ).assign_attrs(norm=Quantity(1, magnetic_geometry.x_point_units))
        self.x_point_z = xr.DataArray(
            np.atleast_1d(magnetic_geometry.x_point_Z)
        ).assign_attrs(norm=Quantity(1, magnetic_geometry.x_point_units))

        self.field_on_axis = xr.DataArray(magnetic_geometry.axis_Btor).assign_attrs(
            norm=Quantity(1, magnetic_geometry.axis_Btor_units)
        )

        # Read the basis vectors for the spline (in normalised units)
        self.spline_basis_r = xr.DataArray(magnetic_geometry.R).assign_attrs(norm="R0")
        self.spline_basis_z = xr.DataArray(magnetic_geometry.Z).assign_attrs(norm="R0")

        # Read the psi data (in Weber)
        try:
            self.psi_units = self.magnetic_geometry.psi_units
        except AttributeError:
            self.psi_units = self.magnetic_geometry.psi.units

        self.psi_data = xr.DataArray(self.magnetic_geometry.psi).assign_attrs(
            norm=Quantity(1, self.psi_units)
        )

        # Note that the axis ordering is inverted relative to the output of
        # meshgrid.
        self.psi_interpolator = RectBivariateSpline(
            self.spline_basis_r, self.spline_basis_z, self.psi_data.T
        )

        if flip_z:
            self.flip_z()

    def flip_z(self):
        """
        Flips the vertical direction of the equilibrium, to affect a toroidal field reversal
        """

        self.flipped_z = not self.flipped_z
        self.poloidal_field_factor *= -1.0

        self.axis_z *= -1.0
        self.x_point_z *= -1.0

        self.spline_basis_z = -self.spline_basis_z[::-1]
        self.psi_data = self.psi_data[::-1, :]

        self.psi_interpolator = RectBivariateSpline(
            self.spline_basis_r, self.spline_basis_z, self.psi_data.T
        )

    @property
    def poloidal_flux_on_axis(self):
        """Poloidal flux in Wb on axis"""
        return xr.DataArray(self.psi_limits.psi_axis).assign_attrs(
            norm=Quantity(1, self.psi_units)
        )

    @property
    def poloidal_flux_on_separatrix(self):
        """Poloidal flux in Wb on separatrix/at the x-point"""
        return xr.DataArray(self.psi_limits.psi_seperatrix).assign_attrs(
            norm=Quantity(1, self.psi_units)
        )

    @property
    def axis_r_norm(self):
        """Magnetic axis radial position in normalised units (1 by definition)"""
        return xr.DataArray(1).assign_attrs(norm="R0")

    @property
    def axis_z_norm(self):
        """Magnetic axis vertical position in normalised units (usually close to 0)"""
        return xr.DataArray(self.axis_z / self.axis_r).assign_attrs(norm="R0")

    @property
    def x_point_r_norm(self):
        """x-point radial position in normalised units"""
        return xr.DataArray(self.x_point_r / self.axis_r).assign_attrs(norm="R0")

    @property
    def x_point_z_norm(self):
        """x-point vertical position in normalised units"""
        return xr.DataArray(self.x_point_z / self.axis_r).assign_attrs(norm="R0")

    @property
    def axis_btor(self):
        """Returns the on-axis field as a quantity"""
        return convert_xarray_to_quantity(self.field_on_axis)

    def poloidal_flux(self, r_norm, z_norm, grid: bool = True):
        """Returns the poloidal flux in Wb at a point r_norm, z_norm"""

        return array_coords(
            xr.DataArray(
                self.psi_interpolator(x=r_norm, y=z_norm, grid=grid).T
            ).assign_attrs(norm=Quantity(1, self.psi_units)),
            grid=grid,
            r_norm=r_norm,
            z_norm=z_norm,
        )

    def magnetic_field_r(self, r_norm, z_norm, grid: bool = True):
        """
        Returns the normalised radial magnetic field at a point r_norm, z_norm

        Evaluate the poloidal flux, taking the 0th x derivative and the 1st y derivative
        Then calculate the radial magnetic field according to eqn 2.25 of H. Zohm, 'MHD stability of Tokamaks', p24

        N.b. One factor of self%axis_r comes from evaluating the derivative on the (r_norm', z_norm') normalised coordinate
        and the second comes from using normalised r_norm', z_norm' for the axis
        """
        psi_dz = xr.DataArray(
            np.atleast_1d(
                self.psi_interpolator(x=r_norm, y=z_norm, dx=0, dy=1, grid=grid).T
            )
        ).assign_attrs(norm=Quantity(1, "Wb"))

        b_r = psi_dz / (
            2
            * np.pi
            * (r_norm[None, ...] if grid else r_norm)
            * self.axis_r.values
            * self.axis_r.values
        )

        return array_coords(
            xr.DataArray(
                -self.poloidal_field_factor * b_r / self.field_on_axis
            ).assign_attrs(norm=self.axis_btor),
            grid=grid,
            r_norm=r_norm,
            z_norm=z_norm,
        )

    def magnetic_field_z(self, r_norm, z_norm, grid: bool = True):
        """
        Returns the normalised vertical magnetic field at a point r_norm, z_norm

        Evaluate the poloidal flux, taking the 1st x derivative and the 0th y derivative
        Then calculate the radial magnetic field according to eqn 2.24 of H. Zohm, 'MHD stability of Tokamaks', p24

        N.b. One factor of self%axis_r comes from evaluating the derivative on the (r_norm', z_norm') normalised coordinate
        and the second comes from using normalised r_norm', z_norm' for the axis
        """
        psi_dr = xr.DataArray(
            np.atleast_1d(
                self.psi_interpolator(x=r_norm, y=z_norm, dx=1, dy=0, grid=grid).T
            )
        ).assign_attrs(norm=Quantity(1, "Wb"))

        b_z = psi_dr / (
            2
            * np.pi
            * (r_norm[None, ...] if grid else r_norm)
            * self.axis_r.values
            * self.axis_r.values
        )

        return array_coords(
            xr.DataArray(
                self.poloidal_field_factor * b_z / self.field_on_axis
            ).assign_attrs(norm=self.axis_btor),
            grid=grid,
            r_norm=r_norm,
            z_norm=z_norm,
        )

    def magnetic_field_toroidal(
        self, r_norm, z_norm, grid: bool = False
    ) -> xr.DataArray:
        """Normalised magnetic field strength in the phi direction"""

        if not grid:
            # Scalar case
            return array_coords(
                xr.DataArray(1.0 / np.atleast_1d(r_norm)).assign_attrs(norm="B0"),
                grid=grid,
                r_norm=r_norm,
                z_norm=z_norm,
            )
        else:
            # Basis vector case
            r_mesh, _ = np.meshgrid(r_norm, z_norm)
            return array_coords(
                xr.DataArray(1.0 / np.atleast_1d(r_mesh)).assign_attrs(norm="B0"),
                grid=grid,
                r_norm=r_norm,
                z_norm=z_norm,
            )

    def magnetic_field_poloidal(
        self, r_norm, z_norm, grid: bool = False
    ) -> xr.DataArray:
        """Normalised magnetic field strength in the poloidal plane. N.b. norm is inherited from components"""
        b_x = self.magnetic_field_r(r_norm, z_norm, grid)
        b_y = self.magnetic_field_z(r_norm, z_norm, grid)
        return xr.DataArray(np.sqrt(b_x ** 2 + b_y ** 2)).assign_attrs(
            norm=self.axis_btor
        )

    def magnetic_field_absolute(
        self, r_norm, z_norm, grid: bool = False
    ) -> xr.DataArray:
        """Total normalised magnetic field strength. N.b. norm is inherited from components"""
        b_x = self.magnetic_field_r(r_norm, z_norm, grid)
        b_y = self.magnetic_field_z(r_norm, z_norm, grid)
        b_t = self.magnetic_field_toroidal(r_norm, z_norm, grid)
        return xr.DataArray(np.sqrt(b_x ** 2 + b_y ** 2 + b_t ** 2)).assign_attrs(
            norm=self.axis_btor
        )

    def normalised_flux_surface_label(
        self, r_norm, z_norm, grid: bool = True
    ) -> xr.DataArray:
        """
        Flux surface which is 0 at the magnetic axis and 1 at the separatrix
        """

        return array_coords(
            self.normalise_flux_surface_label(
                psi_value=self.poloidal_flux(r_norm, z_norm, grid)
            ),
            grid=grid,
            r_norm=r_norm,
            z_norm=z_norm,
        )

    def normalise_flux_surface_label(self, psi_value):
        """
        Provides a method to convert from poloidal flux (psi) to the normalised flux coordinate (rho)
        """
        rho_squared = (psi_value - self.poloidal_flux_on_axis) / (
            self.poloidal_flux_on_separatrix - self.poloidal_flux_on_axis
        )

        rho_squared = np.atleast_1d(rho_squared)
        rho_squared[np.asarray(rho_squared < 0.0)] = 0.0

        return xr.DataArray(np.sqrt(rho_squared)).assign_attrs(norm=Dimensionless)

    def get_separatrix(self, r_norm, z_norm, level: float = 1.0) -> list:
        """
        Returns the separatrix contour

        Can plot a specific contour via plt.plot(*separatrix[index].T) where index is a contour index
        """
        assert r_norm.ndim == 1 and z_norm.ndim == 1

        return find_contours(
            r_norm,
            z_norm,
            self.normalised_flux_surface_label(r_norm, z_norm),
            level=level,
        )

    def poloidal_unit_vector(self, r_norm, z_norm, grid: bool = False) -> xr.DataArray:
        """Unit vector pointing in the direction of the poloidal magnetic field. i.e. 'along a flux surface'"""
        b_x = self.magnetic_field_r(r_norm, z_norm, grid)
        b_y = self.magnetic_field_z(r_norm, z_norm, grid)
        b_pol = np.sqrt(b_x ** 2 + b_y ** 2)

        if grid:
            return poloidal_vector(
                b_x / b_pol,
                b_y / b_pol,
                dims=["Z", "R"],
                coords={"R": r_norm, "Z": z_norm},
            )
        elif r_norm.size == 1:
            return poloidal_vector(b_x / b_pol, b_y / b_pol, dims=[])
        else:
            return poloidal_vector(b_x / b_pol, b_y / b_pol, dims=["points"])

    def radial_unit_vector(self, r_norm, z_norm, grid: bool = False) -> xr.DataArray:
        """Unit vector pointing normal to the direction of the poloidal magnetic field. i.e. 'across a flux surface'"""
        b_x = self.magnetic_field_r(r_norm, z_norm, grid)
        b_y = self.magnetic_field_z(r_norm, z_norm, grid)
        b_pol = np.sqrt(b_x ** 2 + b_y ** 2)

        if grid:
            return poloidal_vector(
                -b_y / b_pol,
                b_x / b_pol,
                dims=["Z", "R"],
                coords={"R": r_norm, "Z": z_norm},
            )
        elif r_norm.size == 1:
            return poloidal_vector(-b_y / b_pol, b_x / b_pol, dims=[])
        else:
            return poloidal_vector(-b_y / b_pol, b_x / b_pol, dims=["points"])

    def parallel_unit_vector(self, r_norm, z_norm, grid: bool = False) -> xr.DataArray:
        """Unit vector pointing in the direction of the total magnetic field"""
        b_x = self.magnetic_field_r(r_norm, z_norm, grid)
        b_y = self.magnetic_field_z(r_norm, z_norm, grid)
        b_t = self.magnetic_field_toroidal(r_norm, z_norm, grid)
        b_abs = np.sqrt(b_x ** 2 + b_y ** 2 + b_t ** 2)

        if grid:
            return cylindrical_vector(
                input_r=b_x / b_abs,
                input_phi=b_t / b_abs,
                input_z=b_y / b_abs,
                dims=["Z", "R"],
                coords={"R": r_norm, "Z": z_norm},
            )
        elif r_norm.size == 1:
            return cylindrical_vector(
                input_r=b_x / b_abs, input_phi=b_t / b_abs, input_z=b_y / b_abs, dims=[]
            )
        else:
            return cylindrical_vector(
                input_r=b_x / b_abs,
                input_phi=b_t / b_abs,
                input_z=b_y / b_abs,
                dims=["points"],
            )
