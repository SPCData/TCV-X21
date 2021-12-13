"""
Converts from rho to R^u radial distance to separatrix and back
"""
from scipy.interpolate import make_interp_spline
import xarray as xr

from .lineouts_m import outboard_midplane_chord
from tcvx21.units_m import Quantity, Dimensionless


class OutboardMidplaneMap:
    def __init__(self, grid, equi, norm):
        """
        Builds an interpolator which uses the flux surface label to translate the flux surface label along an arbitrary line
        to the equivalent distance to the separatrix, if the line was at the outboard midplane.

        This is useful for accounting for flux expansion when comparing fall-off lengths at different positions
        """

        lineout = outboard_midplane_chord(grid, equi)
        self.rho_points = equi.normalised_flux_surface_label(
            lineout.r_points, lineout.z_points, grid=False
        )
        self.arc_points = lineout.poloidal_arc_length(norm=norm)

        omp_map = make_interp_spline(self.rho_points, self.arc_points)
        # Call again to make the separatrix have a value of '0.0'
        self.separatrix_arc = omp_map(1.0)
        omp_map = make_interp_spline(
            self.rho_points, self.arc_points - self.separatrix_arc
        )

        self.equi = equi
        self.omp_map = omp_map

    def __call__(self, r_points, z_points):
        """
        Uses the flux surface label to map the points back to the outboard midplane, and then calculates the equivalent distance
        to the separatrix.

        If the curve includes points in the private flux region, the rho < 1.0 (private flux region) or equivalently
        omp_mapped_distance < 0.0 is mapped to the core, which may have significantly different flux-surface expansion
        """
        rho_points = self.equi.normalised_flux_surface_label(
            r_points, z_points, grid=False
        )
        return self.convert_rho_to_distance(rho_points=rho_points)

    def convert_rho_to_distance(self, rho_points):
        """
        Uses the flux surface label to map an array of normalised flux surface label to outboard-midplane-equivalent distance
        to the separatrix.
        """
        return xr.DataArray(self.omp_map(rho_points)).assign_attrs(
            norm=self.arc_points.norm, name="OMP-mapped distance to sep."
        )

    def convert_distance_to_rho(self, arc_points: Quantity):
        """
        Inverts the omp_map
        """
        omp_inv = make_interp_spline(
            self.arc_points - self.separatrix_arc, self.rho_points
        )

        arc_norm = (arc_points / self.arc_points.norm).to("")

        return xr.DataArray(omp_inv(arc_norm)).assign_attrs(
            norm=Dimensionless, name="Normalised flux-surface label"
        )
