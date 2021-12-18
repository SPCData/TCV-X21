import pytest
import matplotlib.pyplot as plt
import tcvx21
from tcvx21 import plotting, Quantity
import numpy as np


@pytest.fixture(scope="module")
def expt(tcv_forward_data):
    return tcvx21.Record(tcv_forward_data, color="C0", label="TCV", linestyle="-")


@pytest.fixture(scope="module")
def sim_sample():
    return tcvx21.Record(
        tcvx21.test_dir / 'comparison_data' / "GRILLIX_example.nc", color="C1", linestyle="--"
    )


def test_plot_array_as_transparency():

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    x = np.linspace(-5, 5, num=30)
    y = np.linspace(0, 5, num=50)
    x_mesh, y_mesh = np.meshgrid(x, y)

    plt.contour(x, y, np.sin(np.sqrt(x_mesh ** 2 + y_mesh ** 2)))
    im = plotting.plot_array_as_transparency(
        ax,
        x,
        y,
        np.sin(np.sqrt(x_mesh ** 2 + y_mesh ** 2)),
        cmap=plt.cm.Reds,
        intensity=0.5,
    )
    im.set_zorder(np.inf)

    plotting.savefig(fig, tcvx21.test_figures_dir / "array_as_transparency.png")


def test_plot_1D_region(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("LFS-LP", "density")
    obs.plot(ax=ax)

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot1d_region.png")


def test_plot_1D_labels(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("LFS-LP", "density")
    obs.plot(ax=ax)
    plotting.add_x_zero_line(ax)
    plotting.add_y_zero_line(ax)
    plotting.label_subplots(np.array([ax]))
    plotting.add_twinx_label(ax, "Twinx label")

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot1d_labels.png")


def test_plot_1D_errorbar(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("LFS-LP", "density")
    obs.plot(ax=ax, plot_type="errorbar")

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot1d_errorbar.png")


def test_plot_2D_values(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("RDPA", "density")
    obs.plot(ax=ax)

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot2d_values.png")


def test_plot_2D_logvalues(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("RDPA", "density")
    obs.plot(ax=ax, log_cbar=True)

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot2d_logvalues.png")


def test_plot_2D_errors(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("RDPA", "density")
    obs.plot(ax=ax, plot_type="errors")

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot2d_errors.png")


def test_plot_2D_sample_points(sim_sample):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("RDPA", "density")
    obs.plot(ax=ax, plot_type="sample_points")

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot2d_sample_points.png")


def test_plot_2d_interpolation(sim_sample, expt):

    plt.style.use(tcvx21.style_sheet)
    fig, ax = plt.subplots()

    obs = sim_sample.get_observable("RDPA", "density")
    ref = expt.get_observable("RDPA", "density")

    obs.interpolate_onto_reference(ref, plot_comparison=True)

    plotting.savefig(fig, tcvx21.test_figures_dir / "plot2d_interpolation.png")


def test_plot_comparison_1D(sim_sample, expt):

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="LFS-LP",
        observable="density",
        experimental_data={"forward_field": expt},
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot1d_comparison.png",
    )


def test_plot_comparison_1D_simulation_only(sim_sample, expt):

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="LFS-LP",
        observable="ion_temp",
        experimental_data={"forward_field": expt},
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot1d_comparison_simulation_only.png",
    )


def test_plot_comparison_1D_masked(sim_sample, tcv_forward_data):

    expt_with_mask = {
        "forward_field": tcvx21.Record(
            tcv_forward_data, color="C0", label="TCV", linestyle="-"
        )
    }
    expt_with_mask["forward_field"].get_observable("LFS-LP", "density").set_mask(
        position_min=Quantity(-0.9, "cm"), position_max=Quantity(2.5, "cm")
    )

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="LFS-LP",
        observable="density",
        experimental_data=expt_with_mask,
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot1d_comparison_masked.png",
    )


def test_plot_comparison_1D_plotlimits(expt):

    sim_with_limits = {
        "test": {
            "forward_field": tcvx21.Record(
                tcvx21.test_dir / 'comparison_data' / "GRILLIX_example.nc", color="C1", linestyle="--"
            )
        }
    }
    sim_with_limits["test"]["forward_field"].get_observable(
        "LFS-LP", "jsat_kurtosis"
    ).set_plot_limits(ymin=0, ymax=10)

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="LFS-LP",
        observable="density",
        experimental_data={"forward_field": expt},
        simulation_data=sim_with_limits,
        output_path=tcvx21.test_figures_dir / "plot1d_comparison_plotlimits.png",
    )


def test_plot_comparison_2D(sim_sample, expt):

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="RDPA",
        observable="density",
        experimental_data={"forward_field": expt},
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot2d_comparison.png",
    )


def test_plot_comparison_2D_simulation_only(sim_sample, expt):

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="RDPA",
        observable="ion_temp",
        experimental_data={"forward_field": expt},
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot2d_comparison_simulation_only.png",
    )


def test_plot_comparison_2D_no_common_cbar(sim_sample, expt):

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="RDPA",
        observable="density",
        experimental_data={"forward_field": expt},
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot2d_comparison_no_common_cbar.png",
        common_colorbar=False,
    )


def test_plot_comparison_2D_fullrange_cbar(sim_sample, expt):

    plotting.plot_comparison(
        field_direction="forward_field",
        diagnostic="RDPA",
        observable="mach_number",
        experimental_data={"forward_field": expt},
        simulation_data={"test": {"forward_field": sim_sample}},
        output_path=tcvx21.test_figures_dir / "plot2d_comparison_fullrange_cbar.png",
        experiment_sets_colorbar=False,
        diverging=True,
    )


def test_tile1d_diagnostics(comparison_data):

    plotting.tile1d(
        **comparison_data,
        diagnostics=(("FHRP", "TS", "LFS-LP", "HFS-LP")),
        observables=("density"),
        output_path=tcvx21.test_figures_dir / "tile1d_diagnostics.png"
    )


def test_tile1d_observables(comparison_data):

    plotting.tile1d(
        **comparison_data,
        diagnostics="FHRP",
        observables=("density", "electron_temp", "potential", "mach_number"),
        overplot=(("density", "TS", "C4"), ("electron_temp", "TS", "C4")),
        manual_title="Outboard midplane and divertor entrance",
        output_path=tcvx21.test_figures_dir / "tile1d_observables.png"
    )


def test_tile2d_single_observable(comparison_data):

    plotting.tile2d_single_observable(
        **comparison_data,
        diagnostic="RDPA",
        observable="density",
        output_path=tcvx21.test_figures_dir / "tile2d_single_observable.png",
        offset=1e18
    )


def test_tile2d(comparison_data):

    plotting.tile2d(
        **comparison_data,
        diagnostics=("RDPA", "RDPA", "RDPA"),
        observables=("density", "mach_number", "jsat_fluct"),
        labels=("density", "Mach number", "Fluct. of $J_{sat}$"),
        offsets=[1e18, None, None],
        fig_height_per_row=1.5,
        extra_args=(
            dict(),
            dict(diverging=True, cmap="RdBu_r", cbar_lim_=[-1.2, 1.2]),
            dict(log_cbar=False, cbar_lim_=[0, 1.0]),
        ),
        subplots_kwargs=dict(hspace=0.05, wspace=0.05),
        n_contours=11,
        output_path=tcvx21.test_figures_dir / "tile2d.png"
    )
