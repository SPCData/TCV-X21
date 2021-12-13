"""
A class to handle tracing along field-lines
"""
from scipy.integrate import solve_ivp
import numpy as np


class FieldlineTracer:
    """
    Routines to parallel-trace along a fieldline, based on the DOP853 integrator
    """

    def __init__(self, equi, rtol: float = 1e-3, atol: float = 1e-6):
        """
        Simple initialiser
        equi should be an object which defines the magnetic field
        rtol and atol are used to set integration tolerances, for the relative and absolute tolerance
        """
        self.equi = equi
        self.rtol, self.atol = rtol, atol

    def integrate_ivp(self, func, initial, t_max, events=None):
        """Interface to DOP853 integrator"""
        initial = np.array(initial)  # Throw a warning about ragged arrays early

        solution = solve_ivp(
            fun=func,
            t_span=(0.0, t_max),
            y0=initial,
            rtol=self.rtol,
            atol=self.atol,
            method="RK45",
            events=events,
        )

        return solution

    def toroidal_integration(
        self, r_initial, z_initial, max_toroidal_trace, events=None
    ):
        """
        Performs an integration along a magnetic field line, with phi (toroidal angle) as the independent variable

        Returns the (x,y) positions of the trace, as well as the fieldline length (in radians)

        To get the x, y, L_parallel values in physical units, multiply by R0
        """
        r_initial, z_initial = np.squeeze(r_initial), np.squeeze(z_initial)

        solution = self.integrate_ivp(
            func=self.toroidal_integration_equation,
            initial=[r_initial, z_initial, 0.0],
            t_max=max_toroidal_trace,
            events=events,
        )

        return solution

    def toroidal_integration_equation(self, _, state):
        """
        Integrate around the torus (independent variable = toroidal angle)

        The fieldline length is returned as the third element of the state vector
        """

        r_norm = state[0]
        z_norm = state[1]

        b_r = self.equi.magnetic_field_r(r_norm, z_norm, grid=False)
        b_z = self.equi.magnetic_field_z(r_norm, z_norm, grid=False)
        b_t = self.equi.magnetic_field_toroidal(r_norm, z_norm, grid=False)
        jacobian = r_norm

        d_state = np.zeros_like(state)
        d_state[0] = b_r / b_t * jacobian
        d_state[1] = b_z / b_t * jacobian
        d_state[2] = (
            np.sqrt(b_r * b_r / (b_t * b_t) + b_z * b_z / (b_t * b_t) + 1.0) * jacobian
        )

        return d_state

    def find_neighbouring_points(
        self, r_initial, z_initial, n_toroidal_planes: int = 16
    ):
        """
        Finds the neighbours in both directions of an array of sample points
        Although this loop should be trivially parallel, dask parallelism doesn't seem to work here
        """
        r_initial = np.atleast_1d(r_initial)
        z_initial = np.atleast_1d(z_initial)

        assert r_initial.shape == z_initial.shape
        assert (
            r_initial.ndim == 1 and z_initial.ndim == 1
        ), f"Should provide r and z as 1D arrays"

        forward_trace = np.zeros((r_initial.size, 3))
        reverse_trace = np.zeros((r_initial.size, 3))

        print("Tracing", end=" ")
        for i in range(r_initial.size):
            print(f"{i}/{r_initial.size}", end=", ")

            r_in, z_in = r_initial[i], z_initial[i]
            forward_trace[i, :] = self.toroidal_integration(
                r_in, z_in, +2.0 * np.pi / n_toroidal_planes
            ).y[:, -1]
            reverse_trace[i, :] = self.toroidal_integration(
                r_in, z_in, -2.0 * np.pi / n_toroidal_planes
            ).y[:, -1]
        print("Done")

        return forward_trace, reverse_trace
