"""
Testing the post-processing routines included in tcvx21/grillix_post

The tests have two purposes
1. Make sure that you haven't broken anything when editing. For this, a high coverage is useful, even if the
tests aren't particularly meaningful
2. Make sure that the functionality is correct. This is much more difficult -- need to be able to find a
case which you know the right answer to (independently of the diagnostic). For difficult terms like the integrations
and the vector analysis, I've tried to include functional tests

"""
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import tcvx21
from tcvx21.grillix_post import (
    components,
    lineouts,
    observables,
    filepath_resolver,
    WorkFileWriter,
    convert_work_file_to_validation_netcdf,
)
from tcvx21.grillix_post.components import vector_m as vector
from tcvx21 import Quantity, Dimensionless


@pytest.fixture(scope="module")
def file_path():
    return tcvx21.sample_data


@pytest.fixture(scope="module")
def flip_z():
    return True


@pytest.fixture(scope="module")
def grid(file_path):
    return components.Grid(grid_file=filepath_resolver(file_path, "vgrid.nc"))


@pytest.fixture(scope="module")
def norm(file_path):
    return components.Normalisation.initialise_from_normalisation_file(
        filepath_resolver(file_path, "physical_parameters.nml")
    )


@pytest.fixture(scope="module")
def snaps(file_path, norm):
    return components.read_snaps_from_file(
        file_path, norm, time_slice=slice(-1, None), all_planes=True
    )


@pytest.fixture(scope="module")
def snaps_single(file_path, norm):
    return components.read_snaps_from_file(
        file_path, norm, time_slice=slice(None), all_planes=False
    )


@pytest.fixture(scope="module")
def equi(file_path, flip_z):
    return components.Equi(
        filepath_resolver(file_path, "TCV_ortho.nc"),
        filepath_resolver(file_path, "pen_metainfo.nc"),
        flip_z=flip_z,
    )


@pytest.fixture(scope="module")
def params(file_path):
    parameter_filepath = filepath_resolver(file_path, "params.in")
    return components.convert_params_filepaths(
        parameter_filepath, components.read_fortran_namelist(parameter_filepath)
    )


@pytest.fixture(scope="module")
def omp(grid, equi):
    return lineouts.outboard_midplane_chord(grid, equi)


@pytest.fixture(scope="module")
def omp_map(grid, equi, norm):
    return lineouts.OutboardMidplaneMap(grid, equi, norm)


def test_omp_map(omp, omp_map, norm):
    assert np.allclose(
        omp_map(omp.r_points, omp.z_points),
        omp.poloidal_arc_length(norm) - omp_map.separatrix_arc,
    )
    assert np.isclose(omp_map.separatrix_arc, 0.20940273)


@pytest.fixture(scope="module")
def lfs(grid, equi, norm, params, file_path):
    lineout = lineouts.penalisation_contour(
        grid, equi, level=0.0, contour_index=0, n_points=100
    )
    observables.initialise_lineout_for_parallel_gradient(
        lineout,
        grid,
        equi,
        norm,
        params["params_grid"]["npol"],
        stored_trace=file_path / "lfs_trace_for_testing.nc",
    )
    return lineout


def test_lfs_construction(lfs):
    """Check that the LFS lineout comes out at the right position, and that the forward and reverse traces have been defined"""
    assert np.all((lfs.forward_distance + lfs.reverse_distance) < 1e-3)
    assert np.isclose(lfs.r_points.mean(), 0.9048, atol=1e-3)
    assert np.isclose(lfs.z_points.mean(), 0.7838, atol=1e-3)


@pytest.fixture(scope="module")
def hfs(grid, equi):
    return lineouts.penalisation_contour(
        grid, equi, level=0.0, contour_index=1, n_points=100
    )


def test_hfs_construction(lfs, hfs, flip_z):
    assert lfs.r_points.mean() > hfs.r_points.mean()
    assert lfs.z_points.mean() * (-1 if flip_z else 1) < hfs.z_points.mean() * (
        -1 if flip_z else 1
    )


@pytest.fixture(scope="module")
def ts(grid, equi):
    return lineouts.thomson_scattering(
        grid, equi, tcvx21.thomson_coords_json, n_points=100
    )


def test_ts_contruction(ts):
    assert np.allclose(ts.r_points, 0.99353195)


@pytest.fixture(scope="module")
def rdpa(grid, equi, omp_map, norm):
    return lineouts.rdpa(grid, equi, omp_map, norm, tcvx21.rdpa_coords_json)


def test_rdpa_construction(rdpa):
    assert rdpa.r_points.size == 373
    assert np.isclose(rdpa.r_points.mean(), 0.8965, atol=1e-3)


@pytest.fixture(scope="module")
def xpt(grid, equi, norm):
    return lineouts.xpoint(grid, equi, norm, r_samples=10, z_samples=10)


def test_xpt_construction(xpt, equi):
    assert np.isclose(xpt.r_points.mean(), equi.x_point_r_norm, atol=1e-2)
    assert np.isclose(xpt.z_points.mean(), equi.x_point_z_norm, atol=1e-2)


def test_tracing(grid, equi, norm, params, tmpdir):
    """
    Tests whether the parallel_gradient calculation works, both for the tracing and reading from store
    Using a short HFS lineout of only 5 points for efficiency
    """

    hfs_short = lineouts.penalisation_contour(
        grid, equi, level=0.0, contour_index=1, n_points=5
    )

    observables.initialise_lineout_for_parallel_gradient(
        hfs_short,
        grid,
        equi,
        norm,
        params["params_grid"]["npol"],
        stored_trace=Path(tmpdir / "hfs_trace.nc"),
    )

    observables.initialise_lineout_for_parallel_gradient(
        hfs_short,
        grid,
        equi,
        norm,
        params["params_grid"]["npol"],
        stored_trace=Path(tmpdir / "hfs_trace.nc"),
    )


def test_integration(grid, norm):
    """
    Tests whether the poloidal and cylindrical integrations return the analytical solution for the surface of a circle
    (for the poloidal_integration) and for the volume of a torus (for the axisymmetric_cylindrical_integration)
    """

    r_mesh, z_mesh = np.meshgrid(grid.r_s, grid.z_s)

    r_centre, z_centre = 0.8, 0.125
    radius = 0.2
    radius_mesh = np.sqrt((r_mesh - r_centre) ** 2 + (z_mesh - z_centre) ** 2)

    # Make a function which has a circle of radius 0.2 centred at r_centre, z_centre
    test_function = np.zeros_like(r_mesh)
    test_function[radius_mesh < radius] = 1.0
    test_function = xr.DataArray(test_function)
    test_function.attrs["norm"] = Dimensionless

    circle_area_numerical = observables.integrations_m.poloidal_integration(
        grid, norm, test_function
    )
    circle_area_analytical = np.pi * (radius * norm.R0) ** 2
    assert np.isclose(circle_area_numerical, circle_area_analytical, rtol=1e-3)

    torus_volume_numerical = (
        observables.integrations_m.axisymmetric_cylindrical_integration(
            grid, norm, test_function
        )
    )
    torus_volume_analytical = (
        np.pi * (radius * norm.R0) ** 2 * 2.0 * np.pi * (r_centre * norm.R0)
    )

    assert np.isclose(torus_volume_numerical, torus_volume_analytical, rtol=1e-3)


def test_poloidal_vector_gradient(grid, norm):
    """
    Takes the vector gradient of the meshgrids, which internally calls poloidal_vector

    perpendicular_gradient returns on perpendicular length scale, which is normalised to rho_s0
    grid.r_s and grid.z_s are normalised to R0

    gradient(R/R0) = 1/R0 gradient(R) = 1/rho_s0 rho_s0/R0 gradient(R) = rho_star * gradient(R/rho_s0)

    Would expect gradient(r_mesh) w.r.t. R0 = 1, sp gradient(r_mesh) w.r.t. rho_s0 = 1/rho_star
    """

    # Get the grid in real units
    r_mesh, z_mesh = np.meshgrid(grid.r_s, grid.z_s)
    r_mesh = xr.DataArray(
        r_mesh, dims=["Z", "R"], coords={"R": grid.r_s, "Z": grid.z_s}
    )
    z_mesh = xr.DataArray(
        z_mesh, dims=["Z", "R"], coords={"R": grid.r_s, "Z": grid.z_s}
    )

    # Conversion from parallel to perpendicular length scales
    delta = (norm.R0 / norm.rho_s0).to("").magnitude

    # Try take the perpendicular gradient of the grid values on the perpendicular length scale
    rmesh_gradient_perp_perp = observables.perpendicular_gradient(
        grid, norm, r_mesh * delta
    )
    zmesh_gradient_perp_perp = observables.perpendicular_gradient(
        grid, norm, z_mesh * delta
    )

    assert np.allclose(rmesh_gradient_perp_perp.sel(vector="eR"), np.ones_like(r_mesh))
    assert np.allclose(
        rmesh_gradient_perp_perp.sel(vector="ePhi"), np.zeros_like(r_mesh)
    )
    assert np.allclose(rmesh_gradient_perp_perp.sel(vector="eZ"), np.zeros_like(r_mesh))

    assert np.allclose(zmesh_gradient_perp_perp.sel(vector="eR"), np.zeros_like(z_mesh))
    assert np.allclose(
        zmesh_gradient_perp_perp.sel(vector="ePhi"), np.zeros_like(z_mesh)
    )
    assert np.allclose(zmesh_gradient_perp_perp.sel(vector="eZ"), np.ones_like(z_mesh))

    assert rmesh_gradient_perp_perp.norm == 1.0 / norm.rho_s0

    # Try take the perpendicular gradient normalised to the parallel scale length of the grid values (normalised to the parallel
    # scale length)
    rmesh_gradient_par_par = observables.perpendicular_gradient(
        grid, norm, r_mesh, normalised_to_Lperp=False
    )
    zmesh_gradient_par_par = observables.perpendicular_gradient(
        grid, norm, z_mesh, normalised_to_Lperp=False
    )

    assert np.allclose(rmesh_gradient_par_par.sel(vector="eR"), np.ones_like(r_mesh))
    assert np.allclose(rmesh_gradient_par_par.sel(vector="ePhi"), np.zeros_like(r_mesh))
    assert np.allclose(rmesh_gradient_par_par.sel(vector="eZ"), np.zeros_like(r_mesh))

    assert np.allclose(zmesh_gradient_par_par.sel(vector="eR"), np.zeros_like(z_mesh))
    assert np.allclose(zmesh_gradient_par_par.sel(vector="ePhi"), np.zeros_like(z_mesh))
    assert np.allclose(zmesh_gradient_par_par.sel(vector="eZ"), np.ones_like(z_mesh))

    assert rmesh_gradient_par_par.norm == 1.0 / norm.R0

    # See if the SI units come out the same
    rmesh_gradient_par_perp = observables.perpendicular_gradient(grid, norm, r_mesh)
    zmesh_gradient_par_perp = observables.perpendicular_gradient(grid, norm, z_mesh)

    print(
        np.mean(
            rmesh_gradient_par_par.values
            * rmesh_gradient_par_par.norm.to("1/m").magnitude
        )
    )
    print(
        np.mean(
            rmesh_gradient_par_perp.values
            * rmesh_gradient_par_perp.norm.to("1/m").magnitude
        )
    )

    assert np.allclose(
        rmesh_gradient_par_par.values * rmesh_gradient_par_par.norm.to("1/m").magnitude,
        rmesh_gradient_par_perp.values
        * rmesh_gradient_par_perp.norm.to("1/m").magnitude,
    )

    assert np.allclose(
        zmesh_gradient_par_par.values * zmesh_gradient_par_par.norm.to("1/m").magnitude,
        zmesh_gradient_par_perp.values
        * zmesh_gradient_par_perp.norm.to("1/m").magnitude,
    )


def test_toroidal_vector(grid, equi):
    """
    Checks whether a toroidal vector representing Btor behaves as expected
    """
    grid, equi = grid, equi

    btor = equi.magnetic_field_toroidal(grid.r_u, grid.z_u)

    # magnetic_field_toroidal default returns as a scalar. Use vector.toroidal_vector to convert to a vector
    btor_vector = vector.toroidal_vector(btor)

    # 3 different ways of indexing -- remember that the 'vector' dimension is (eR, ePhi, eZ)

    # Check that the eR component is zero
    np.allclose(np.zeros_like(btor), btor_vector[..., 0])
    # Check that the ePhi component is equal to btor
    np.allclose(btor, btor_vector.sel(vector="ePhi"))
    # Check that the eZ component is zero
    np.allclose(np.zeros_like(btor), btor_vector.isel(vector=2))


def test_vector_operations(grid, equi):
    """
    Tests the equilibrium unit vectors (which return as vector arrays)
    """
    grid, equi = grid, equi

    epol = equi.poloidal_unit_vector(grid.r_u, grid.z_u)
    erad = equi.radial_unit_vector(grid.r_u, grid.z_u)
    epar = equi.parallel_unit_vector(grid.r_u, grid.z_u)

    # Can actually use the scalar quantities -- since the unit vectors are constant in their coordinate system
    # However, the grid quantities are more useful since they can promote scalars to vector fields
    eR = vector.eR_unit_vector(grid.r_u, grid.z_u)
    ePhi = vector.ePhi_unit_vector(grid.r_u, grid.z_u)
    eZ = vector.eZ_unit_vector(grid.r_u, grid.z_u)

    # Check that the unit vectors have unit magnitude
    assert np.allclose(vector.vector_magnitude(epol), 1.0)
    assert np.allclose(vector.vector_magnitude(erad), 1.0)
    assert np.allclose(vector.vector_magnitude(epar), 1.0)

    # epol, erad and etor should be mutually orthogonal (dot product zero)
    assert np.allclose(vector.vector_dot(epol, erad), 0.0)
    assert np.allclose(vector.vector_dot(epol, ePhi), 0.0)

    # Right handed coordinate system, so
    assert np.allclose(vector.vector_cross(eR, ePhi), eZ)
    assert np.allclose(vector.vector_cross(ePhi, eZ), eR)
    assert np.allclose(vector.vector_cross(eZ, eR), ePhi)
    # Also, there is the flux-coordinate vector basis (n.b. the parallel unit vector is non-orthogonal)
    assert np.allclose(vector.vector_cross(epol, ePhi), erad)
    assert np.allclose(vector.vector_cross(ePhi, erad), epol)
    assert np.allclose(vector.vector_cross(erad, epol), ePhi)

    # Since epar is a unit vector, the unit vector in the direction of epar should be epar
    assert np.allclose(vector.unit_vector(epar), epar)

    # Check that the dot product is commutitive
    assert np.allclose(vector.vector_dot(epol, epar), vector.vector_dot(epar, epol))

    # The radial unit vector should be perpendicular (in plane) to the parallel unit vector
    assert np.allclose(vector.vector_dot(epar, erad), 0.0)
    # The parallel unit vector projected into the poloidal direction should have magnitude
    # between 0 and 1
    assert np.all(vector.vector_dot(epar, epol) > 0.0)
    assert np.all(vector.vector_dot(epar, epol) < 1.0)


def test_vector_projections(grid, equi):

    grid, equi = grid, equi

    bpol = equi.poloidal_unit_vector(grid.r_u, grid.z_u) * equi.magnetic_field_poloidal(
        grid.r_u, grid.z_u
    )
    bpar = equi.parallel_unit_vector(grid.r_u, grid.z_u) * equi.magnetic_field_absolute(
        grid.r_u, grid.z_u
    )
    btor = vector.toroidal_vector(equi.magnetic_field_toroidal(grid.r_u, grid.z_u))

    # Check that the vector projections match the definitions. N.b. the random scalar factors
    # ensure that it is only the direction (and not the magnitude) of the second argument to vector.vector_projection
    # that actually matters
    assert np.allclose(vector.vector_projection(bpar, 0.57 * btor), btor)
    assert np.allclose(vector.vector_projection(bpar, 0.82 * bpol), bpol)
    assert np.allclose(bpar - vector.vector_rejection(bpar, 0.96 * bpol), bpol)

    # Check that the poloidal vector magnitude actually takes the poloidal component
    assert np.allclose(
        vector.poloidal_vector_magnitude(bpar), vector.vector_magnitude(bpol)
    )
    # Check that the quadrature of the vector magnitude behaves as expected
    assert np.allclose(
        np.sqrt(
            vector.vector_magnitude(bpol) ** 2 + vector.vector_magnitude(btor) ** 2
        ),
        vector.vector_magnitude(bpar),
    )


def test_grid_shape(grid, snaps_single):
    """Order of shaping shouldn't matter"""

    print(grid.shape(snaps_single).isel(R=10, Z=10, tau=-1))


def test_statistical_moments(snaps_single, lfs, norm):
    """Test the computation of the statistical moments"""

    sound_speed = observables.sound_speed(
        electron_temp=snaps_single.electron_temp,
        ion_temp=snaps_single.ion_temp,
        norm=norm,
    )

    jsat = observables.ion_saturation_current(
        density=snaps_single.density,
        sound_speed=sound_speed,
        norm=norm,
        wall_probe=True,
    )

    assert np.isclose(sound_speed.mean(), 1.7886263)
    assert np.isclose(jsat.mean(), 4.38476969)

    lfs_jsat = lfs.interpolate(jsat).rename({"interp_points": "points"})
    assert np.isclose(lfs_jsat.mean(), 0.05644217)

    mean_no_bootstrap = tcvx21.analysis.statistics_m.compute_statistical_moment(
        lfs_jsat, moment="mean"
    )
    std_no_bootstrap = tcvx21.analysis.statistics_m.compute_statistical_moment(
        lfs_jsat, moment="std"
    )
    skew_no_bootstrap = tcvx21.analysis.statistics_m.compute_statistical_moment(
        lfs_jsat, moment="skew"
    )
    kurt_no_bootstrap = tcvx21.analysis.statistics_m.compute_statistical_moment(
        lfs_jsat, moment="kurt"
    )

    (
        jsat_mean,
        jsat_mean_err,
    ) = tcvx21.analysis.compute_statistical_moment_with_bootstrap(
        lfs_jsat, moment="mean", random_seed=12345
    )

    print("mean_no_bootstrap.mean()", mean_no_bootstrap.mean().values)
    print("std_no_bootstrap.mean()", std_no_bootstrap.mean().values)
    print("skew_no_bootstrap.mean()", skew_no_bootstrap.mean().values)
    print("kurt_no_bootstrap.mean()", kurt_no_bootstrap.mean().values)
    print("jsat_mean.mean()", jsat_mean.mean().values)
    print("jsat_mean_err.mean()", jsat_mean_err.mean().values)

    assert np.isclose(mean_no_bootstrap.mean().values, 0.0564421693746107)
    assert np.isclose(std_no_bootstrap.mean().values, 0.000385137413514619)
    assert np.isclose(skew_no_bootstrap.mean().values, 0.012766871897844354)
    assert np.isclose(kurt_no_bootstrap.mean().values, 1.6113993111917615)

    assert np.isclose(
        jsat_mean.mean().values, mean_no_bootstrap.mean().values, rtol=1e-2
    )
    assert np.isclose(jsat_mean.mean().values, 0.056441286051105585, rtol=1e-2)
    assert np.isclose(jsat_mean_err.mean().values, 0.000631788032499611, rtol=1e-2)


def test_heat_flux_fitting(lfs, snaps, equi, norm, omp_map, grid):
    """Test if the heat flux can be calculated and if an Eich function can be fitted"""
    single_snap = snaps.isel(tau=-1)
    single_plane = single_snap.isel(phi=0)

    lineout_rho = equi.normalised_flux_surface_label(
        lfs.r_points, lfs.z_points, grid=False
    )
    lineout_ru = tcvx21.convert_xarray_to_quantity(
        omp_map.convert_rho_to_distance(lineout_rho)
    ).to("cm")

    density = lfs.interpolate(single_plane.density)
    ion_velocity = lfs.interpolate(single_plane.velocity)
    current = lfs.interpolate(single_plane.current)
    electron_temp = lfs.interpolate(single_plane.electron_temp)
    electron_temp_parallel_gradient = observables.compute_gradient_on_plane(
        lfs, single_snap.electron_temp, plane=0
    )
    ion_temp = lfs.interpolate(single_plane.ion_temp)
    ion_temp_parallel_gradient = observables.compute_gradient_on_plane(
        lfs, single_snap.ion_temp, plane=0
    )
    effective_parallel_exb = lfs.interpolate(
        observables.effective_parallel_exb_velocity(
            grid, equi, norm, single_plane.potential
        )
    )

    q_par = observables.total_parallel_heat_flux(
        density,
        electron_temp,
        electron_temp_parallel_gradient,
        ion_temp,
        ion_temp_parallel_gradient,
        ion_velocity,
        current,
        effective_parallel_exb,
        norm,
    ).load()

    _, (lambda_q, lambda_q_err), _, _, _ = tcvx21.analysis.fit_eich_profile(
        lineout_ru, tcvx21.convert_xarray_to_quantity(q_par), plot=False
    )

    assert np.isclose(lambda_q, Quantity(4.15884238, "mm"), rtol=1e-3)
    assert np.isclose(lambda_q_err, Quantity(0.27093345, "mm"), rtol=1e-3)


def test_source_integration(grid, equi, norm, params, snaps_single):
    """Test whether the source integration returns the expected value"""

    integrated_sources = components.integrated_sources(
        grid, equi, norm, params, snaps_single
    )

    assert np.abs(integrated_sources["power_source"] - Quantity(150, "kW")) < Quantity(
        1, "kW"
    )


def test_work_file_writer(file_path, tmpdir):
    """Test the integrated analysis routines"""

    data = WorkFileWriter(
        file_path=file_path,
        work_file=Path(tmpdir / "processing_file.nc"),
        toroidal_field_direction="forward",
        n_points=5,
        data_length=2,
        make_work_file=True,
    )

    data.fill_work_file()

    convert_work_file_to_validation_netcdf(
        work_file=Path(tmpdir / "processing_file.nc"),
        output_file=Path(tmpdir / "GRILLIX_example.nc"),
        simulation_hierarchy=tcvx21.read_from_json(
            tcvx21.sample_data / "simulation_hierarchy.json"
        ),
    )
