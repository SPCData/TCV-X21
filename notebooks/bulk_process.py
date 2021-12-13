# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bulk analysis
#
# This notebook loops over all diagnostic and observables, making the tables and figures provided in the `results` folder.
#
# If you'd like to see the figures plotted inline, you can set `show_figures = True`.

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tcvx21
from tcvx21 import Quantity, test_session

# Apply the custom style sheet, which makes the plots look the same
plt.style.use(tcvx21.style_sheet)

show_figures = False

# %% [markdown]
# ## Data loading
#
# The data is stored in NetCDF files in the `data` folder. The structure of the data files is set by the template file `observables.json`, which allows us to use the same methods for all the data.
#
# Where data is missing, we use an `EmptyRecord` to indicate that the entire comparison should be neglected, or an `EmptyObservable` to indicate that a single observable should be neglected. Both have the property `.is_empty = True`

# %%
from tcvx21 import Record, EmptyRecord

# Experiment reference data
experimental_data = dict(
    forward_field=Record(
        tcvx21.experimental_reference_dir / "TCV_forward_field.nc", color="C0"
    ),
    reversed_field=Record(
        tcvx21.experimental_reference_dir / "TCV_reversed_field.nc", color="C0"
    ),
)
simulation_data = dict(
    GBS=dict(
        forward_field=Record(
            tcvx21.gbs_dir / "GBS_forward_field.nc", color="C2", linestyle="dashed"
        ),
        reversed_field=Record(
            tcvx21.gbs_dir / "GBS_reversed_field.nc", color="C2", linestyle="dashed"
        ),
    ),
    GRILLIX=dict(
        forward_field=Record(
            tcvx21.grillix_dir / "GRILLIX_forward_field.nc",
            color="C1",
            label="GRILLIX",
            linestyle="dotted",
        ),
        reversed_field=Record(
            tcvx21.grillix_dir / "GRILLIX_reversed_field.nc",
            color="C1",
            label="GRILLIX",
            linestyle="dotted",
        ),
    ),
    TOKAM3X=dict(
        forward_field=Record(
            tcvx21.tokam3x_dir / "TOKAM3X_forward_field.nc",
            color="C3",
            label="TOKAM3X",
            linestyle="dashdot",
        ),
        reversed_field=EmptyRecord(),
    ),
)

# Set the error to zero for all simulations (i.e. drop the bootstrapping error from GRILLIX)
for code_results in simulation_data.values():
    for field_direction_results in code_results.values():
        if not field_direction_results.is_empty:
            field_direction_results.set_error_to_zero()


# %% [markdown]
# ## Limiting the $R^u_{sep}$-range of the validation
#
# ### Target Langmuir probes
# The Langmuir probe diagnostic doesn't provide sensible data when the plasma becomes cold and/or
# low-density, since we can't reliably fit IV curves. To exclude these points from our analysis,
# we mark x-limits for the LFS and HFS Langmuir probe arrays.
#
# We select limits where the $J_{sat}$ signal is significantly above the background, and the
# signal as a function of position is reasonably smooth. These conditions are somewhat subjective,
# so you can try select different limits and see how that affects the analysis.
#
# Once we have set `limits`, we can then pass this to `get_observable` and `plot_observable` to
# limit the range of data returned.
#
# ### Other diagnostics
#
# We also crop the other measurements, despite the signal being strong enough to compare to. This is because
# GRILLIX and TOKAM3X use limiting flux-surfaces, and as such don't model the far-SOL.
#
# GBS, on the other hand, does include the far-SOL. Rather than comparing different sets of points between codes,
# we instead select the range of points where all codes provide data (since otherwise the far-SOL comparison might
# unfairly disadvantage GBS in case of disagreement)
#
# We don't, however, crop the `LFS-IR` diagnostic, since the $q_\parallel$ measurements appear reasonable over the available data range, and including the full range helps to determine the $q_\parallel$ background
#
# ## Limiting the $Z_X$-range of the validation
#
# We also want to remove the data very close to the target in the RDPA, because we are combining
# in $Z-Z_X$ coordinates. This means that vertical translation of the plasma will cause
# shifting of the wall in the coordinates we are using. Close to the wall, we expect
# strong gradients due to the sheath, so combining vertically-translated shots might lead
# to combining data which is not comparable. To avoid this, we simply crop our comparison
#

# %%
def set_limits_from_observable(
    experimental_data,
    field_direction,
    diagnostic,
    observable_key="jsat",
    position_min=Quantity(-np.inf, "cm"),
    position_max=Quantity(np.inf, "cm"),
    plot: bool = False,
):
    """
    Sets a mask on the experimental data to limit the range which should be included for analysis
    """

    if plot:
        _, ax = plt.subplots(figsize=(3, 2))
    observable = experimental_data[field_direction].get_observable(
        diagnostic, observable_key
    )
    # Reset the mask to plot full range
    observable.set_mask()

    if plot:
        observable.plot(ax)
        ax.set_title(f"{field_direction}:{diagnostic}:{observable_key}")
        ax.axvline(position_min)
        ax.axvline(position_max)

    observable.set_mask(position_min=position_min, position_max=position_max)
    if plot:
        plt.errorbar(
            observable.positions, observable.values, observable.errors, color="C1"
        )

    for diagnostic_, observable_ in experimental_data[field_direction].keys():
        if diagnostic_ == diagnostic:
            observable = experimental_data[field_direction].get_observable(
                diagnostic_, observable_
            )
            if observable.is_empty:
                continue
            else:
                observable.set_mask(
                    position_min=position_min, position_max=position_max
                )


plot_cropping = False
# Set target crops
set_limits_from_observable(
    experimental_data,
    "forward_field",
    "HFS-LP",
    position_min=Quantity(-0.9, "cm"),
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)
set_limits_from_observable(
    experimental_data,
    "forward_field",
    "LFS-LP",
    position_min=Quantity(-0.9, "cm"),
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)
set_limits_from_observable(
    experimental_data,
    "reversed_field",
    "HFS-LP",
    position_min=Quantity(-0.9, "cm"),
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)
set_limits_from_observable(
    experimental_data,
    "reversed_field",
    "LFS-LP",
    position_min=Quantity(-0.9, "cm"),
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)

set_limits_from_observable(
    experimental_data,
    "forward_field",
    "FHRP",
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)
set_limits_from_observable(
    experimental_data,
    "reversed_field",
    "FHRP",
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)
set_limits_from_observable(
    experimental_data,
    "forward_field",
    "TS",
    "density",
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)
set_limits_from_observable(
    experimental_data,
    "reversed_field",
    "TS",
    "density",
    position_max=Quantity(2.5, "cm"),
    plot=plot_cropping,
)

# Crop the RDPA
for expt in experimental_data.values():
    for diagnostic, observable in expt.keys():
        if not diagnostic == "RDPA":
            continue
        m = expt.get_observable(diagnostic, observable)
        if not m.is_empty:
            m.set_mask(zx_min=Quantity(-0.32, "m"))

# %% [markdown]
# ## Quantitative validation
#
# We start by performing a quantitative validation. The details of this can be found in Oliveira and Body et al, 2021, which are originally from Ricci et al, 2015.

# %%
import pandas as pd
from tcvx21.quant_validation import RicciValidation, write_cases_to_latex

validation_cases = {
    "GBS+": RicciValidation(
        experimental_data["forward_field"], simulation_data["GBS"]["forward_field"]
    ),
    "GBS-": RicciValidation(
        experimental_data["reversed_field"], simulation_data["GBS"]["reversed_field"]
    ),
    "GRILLIX+": RicciValidation(
        experimental_data["forward_field"], simulation_data["GRILLIX"]["forward_field"]
    ),
    "GRILLIX-": RicciValidation(
        experimental_data["reversed_field"],
        simulation_data["GRILLIX"]["reversed_field"],
    ),
    "TOKAM3X+": RicciValidation(
        experimental_data["forward_field"], simulation_data["TOKAM3X"]["forward_field"]
    ),
}

keys, tables = [], []
for key, case in validation_cases.items():
    case.calculate_metric_terms()
    keys.append(key)
    tables.append(case.as_dataframe())
combined_table = pd.concat(tables, axis=1, keys=keys)

# Write the result to a LaTeX table
write_cases_to_latex(
    cases=validation_cases,
    output_file=tcvx21.results_dir / "tables" / "validation_table.tex",
)
# Display a HTML rendering of the validation table
combined_table.style.background_gradient().set_precision(2)

# %% [markdown]
# ## Profile fitting
#
# We also fit profiles into the comparison. Although these don't directly affect the validation $\chi$, they are useful for quantitatively comparing profiles.

# %%
from tcvx21.analysis import make_decay_rate_table, make_eich_fit_comparison

make_decay_rate_table(
    experimental_data,
    simulation_data,
    diagnostics=("FHRP", "TS", "FHRP", "TS"),
    observables=("density", "density", "electron_temp", "electron_temp"),
    labels=("n,OMP", "n,DE", "T_e,OMP", "T_e,DE"),
    fit_range=(0, 1.5),
)

# %%
make_eich_fit_comparison(experimental_data, simulation_data)

# %%
if test_session:
    plt.close("all")

# %% [markdown]
# # Qualitative comparison
#
# We also want to be able to visualise the profiles. We make several figures of the comparison.

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from tcvx21.analysis.statistics_m import strip_moment


def set_plot_limits(m, observable, xmin=-0.9, xmax=2.5):
    base, moment = strip_moment(observable)

    if m.is_empty or m.dimensionality == 2:
        return

    m.set_plot_limits(xmin=xmin, xmax=xmax)

    if "skew" in observable:
        m.set_plot_limits(ymin=-2, ymax=2)
    if "kurtosis" in observable:
        m.set_plot_limits(ymin=0, ymax=8)
    if (
        base in ["density", "electron_temp", "ion_temp", "jsat", "q_parallel"]
        and moment == "mean"
    ):
        m.set_plot_limits(ymin=0)
    if moment == "std":
        m.set_plot_limits(ymin=0)
    if diagnostic == "FHRP" or diagnostic == "TS":
        if "density" == observable:
            m.set_plot_limits(ymax=1e19)
        if "potential" == observable:
            m.set_plot_limits(ymin=-25)
        if "jsat" == observable:
            m.set_plot_limits(ymax=55)
        if "jsat_std" == observable:
            m.set_plot_limits(ymax=10)
        if "jsat_kurtosis" == observable:
            m.set_plot_limits(ymin=0, ymax=15)
        if "vfloat" == observable:
            m.set_plot_limits(ymin=-40, ymax=40)
        if "vfloat_std" == observable:
            m.set_plot_limits(ymax=15)


for field_direction, dataset in experimental_data.items():
    for diagnostic, observable in dataset.keys():
        m = experimental_data[field_direction].get_observable(diagnostic, observable)
        set_plot_limits(m, observable)

for code, simulation_dataset in simulation_data.items():
    for field_direction, dataset in simulation_dataset.items():
        if dataset.is_empty:
            continue

        for diagnostic, observable in dataset.keys():
            if diagnostic == "Xpt":
                continue
            else:
                m = simulation_dataset[field_direction].get_observable(
                    diagnostic, observable
                )
                set_plot_limits(m, observable)

# %%
# Remake the Eich fit plot with the adjusted limits
make_eich_fit_comparison(experimental_data, simulation_data)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"} tags=[]
# Plot each observable individually
from tcvx21.plotting.plot_comparison_m import plot_comparison

for field_direction, dataset in experimental_data.items():
    for diagnostic, observable in dataset.keys():

        plot_comparison(
            field_direction,
            diagnostic,
            observable,
            experimental_data=experimental_data,
            simulation_data=simulation_data,
            show=show_figures,
        )

        if test_session:
            break

    for diagnostic in ["LFS-LP", "HFS-LP", "FHRP", "RDPA"]:

        plot_comparison(
            field_direction,
            diagnostic,
            "ion_temp",
            experimental_data=experimental_data,
            simulation_data=simulation_data,
            show=show_figures,
        )

        if test_session:
            break

if test_session:
    plt.close("all")


# %%
# Plot 'tiled' subplots of 1D observables
from tcvx21.plotting import tile1d

observables_to_plot = [
    dict(observables="density", diagnostics=("FHRP", "TS", "LFS-LP", "HFS-LP")),
    dict(
        observables="electron_temp",
        diagnostics=("FHRP", "TS", "LFS-LP", "HFS-LP"),
        legend_loc=("upper right",),
    ),
    dict(observables="mach_number", diagnostics=("FHRP",)),
    dict(observables="current", diagnostics=("LFS-LP", "HFS-LP")),
    dict(observables="current_std", diagnostics=("LFS-LP", "HFS-LP")),
    dict(observables="q_parallel", diagnostics=("LFS-IR",)),
]

for observable_tiled in observables_to_plot:

    tile1d(
        experimental_data=experimental_data,
        simulation_data=simulation_data,
        **observable_tiled,
        show=show_figures,
    )

    if test_session:
        break

diagnostics_to_plot = [
    dict(
        diagnostics="FHRP",
        observables=("density", "electron_temp", "potential", "mach_number"),
        overplot=(("density", "TS", "C4"), ("electron_temp", "TS", "C4")),
        manual_title="Outboard midplane and divertor entrance",
    ),
    dict(diagnostics="TS", observables=("density", "electron_temp")),
    dict(diagnostics="FHRP", observables=("jsat", "jsat_std", "vfloat", "vfloat_std")),
    dict(
        diagnostics="FHRP",
        observables=("jsat", "jsat_fluct", "jsat_skew", "jsat_kurtosis"),
    ),
    dict(
        diagnostics="LFS-LP",
        observables=("density", "electron_temp", "potential", "current"),
        legend_loc=(
            "upper right",
            "upper right",
            "upper right",
            "upper right",
            "upper right",
            "upper right",
            "lower right",
            "lower right",
        ),
    ),
    dict(
        diagnostics="HFS-LP",
        observables=("density", "electron_temp", "potential", "current"),
    ),
    dict(
        diagnostics="LFS-LP",
        observables=("jsat", "jsat_std", "vfloat", "vfloat_std"),
    ),
    dict(
        diagnostics="HFS-LP",
        observables=("jsat", "jsat_std", "vfloat", "vfloat_std"),
    ),
    dict(
        diagnostics="LFS-LP",
        observables=("jsat", "jsat_fluct", "jsat_skew", "jsat_kurtosis"),
    ),
    dict(
        diagnostics="HFS-LP",
        observables=("jsat", "jsat_fluct", "jsat_skew", "jsat_kurtosis"),
    ),
    dict(diagnostics="LFS-LP", observables=("current", "current_std")),
    dict(diagnostics="HFS-LP", observables=("current", "current_std")),
]

for diagnostic_tiled in diagnostics_to_plot:

    tile1d(
        experimental_data=experimental_data,
        simulation_data=simulation_data,
        **diagnostic_tiled,
        show=show_figures,
    )

    if test_session:
        break

if test_session:
    plt.close("all")

# %% [markdown]
# # Ion temperature analysis
# Plot the ion temperature, with respect to the electron temperature
#
# We don't have data on $T_i$, but can compare it to $T_e$

# %%
from tcvx21.plotting.labels_m import (
    add_twinx_label,
    add_x_zero_line,
    add_y_zero_line,
    make_diagnostic_string,
    make_field_direction_string,
    label_subplots,
    format_yaxis,
)
from tcvx21.plotting.save_figure_m import savefig

fig_width = 7.5
fig_height_per_row = 2.0
title_height = 1.0
nrows = 4
fig, axs = plt.subplots(
    nrows=nrows,
    ncols=2,
    sharex="col",
    sharey="row",
    figsize=(fig_width, title_height + nrows * fig_height_per_row),
    squeeze=False,
)

for column, field_direction in enumerate(["forward_field", "reversed_field"]):
    reference = list(simulation_data.values())[0][field_direction]

    for row, diagnostic in enumerate(["FHRP", "TS", "LFS-LP", "HFS-LP"]):
        ax = axs[row][column]

        for code, case in simulation_data.items():
            case[field_direction].get_observable(diagnostic, "electron_temp").plot(
                ax=ax, linestyle="--", label=f"{code} $T_e$"
            )
            case[field_direction].get_observable(diagnostic, "ion_temp").plot(
                ax=ax, linestyle="-", label=f"{code} $T_i$"
            )

        add_x_zero_line(ax)
        add_y_zero_line(ax)

        ref = reference.get_observable(diagnostic, "ion_temp")
        cases = []
        for fd in ["forward_field", "reversed_field"]:
            cases += [
                case[fd].get_observable(diagnostic, "ion_temp")
                for case in simulation_data.values()
                if not case[fd].is_empty
            ]

        ax.set_ylim(*ref.ylims_in_trim(cases, trim_to_x=ref.xlim))

        reference.get_observable(diagnostic, "ion_temp").apply_plot_limits(ax)

        if row != nrows - 1:
            ax.set_xlabel("")
        if column != 0:
            add_twinx_label(ax, make_diagnostic_string(diagnostic))
        if row == 0:
            ax.set_title(make_field_direction_string(field_direction))

axs[0][0].legend()
_, fig_height = fig.get_size_inches()
plt.subplots_adjust(top=1 - title_height / fig_height)

plt.suptitle(
    "Ion and electron temperatures",
    fontsize="large",
    y=1 - title_height / 2 / fig_height,
)

label_subplots(axs.flatten())

for row in range(nrows):
    format_yaxis(axs[row, 0])
    axs[row, 1].set_ylabel("")

savefig(
    fig,
    output_path=tcvx21.results_dir / "analysis_fig" / f"Ion_temperature.png",
    show=True,
)

if test_session:
    plt.close("all")

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# Plot 'tiled' subplots of 2D observables
from tcvx21.plotting import tile2d_single_observable, tile2d

# Tile both field directions of single 2D observables
rdpa_to_plot = [
    dict(observable="density", offset=1e18),
    dict(observable="electron_temp"),
    dict(observable="potential"),
    dict(
        observable="mach_number",
        cmap="RdBu_r",
        experiment_sets_cbar=False,
        diverging=True,
    ),
    dict(observable="potential", log_cbar=False),
    dict(observable="jsat"),
    dict(observable="jsat_std"),
    dict(observable="jsat_skew"),
    dict(observable="jsat_kurtosis"),
    dict(observable="vfloat"),
    dict(observable="vfloat_std"),
]

for rdpa_tiled in rdpa_to_plot:
    tile2d_single_observable(
        diagnostic="RDPA",
        experimental_data=experimental_data,
        simulation_data=simulation_data,
        **rdpa_tiled,
        show=show_figures,
    )

    if test_session:
        break

if test_session:
    plt.close("all")

# %%
tile2d(
    experimental_data,
    simulation_data,
    ("RDPA", "RDPA", "RDPA", "RDPA"),
    ("density", "potential", "mach_number", "jsat_std"),
    labels=("density", "potential", "Mach number", "Std. dev. of $J_{sat}$"),
    offsets=[1e18, None, None, None],
    fig_height_per_row=1.36,
    extra_args=(
        dict(
            ticks=[0, 1e18, 2e18, 3e18, 4e18, 5e18, 6e18],
            cbar_lim_=Quantity([0, 6e18], "m^-3"),
        ),
        dict(ticks=[0, 20, 40, 60, 80, 100], cbar_lim_=Quantity([0, 90], "V")),
        dict(diverging=True, cmap="RdBu_r", cbar_lim_=[-1.2, 1.2]),
        dict(
            log_cbar=False,
            ticks=[0, 1, 2, 3, 4, 5, 6],
            cbar_lim_=Quantity([0, 6], "kA/m^2"),
        ),
    ),
    subplots_kwargs=dict(hspace=0.05, wspace=0.05),
    n_contours=11,
    show=not (test_session),
)

# %%
tile2d(
    experimental_data,
    simulation_data,
    ("RDPA", "RDPA", "RDPA", "RDPA"),
    ("density", "potential", "mach_number", "jsat_fluct"),
    labels=("density", "potential", "Mach number", "Fluct. of $J_{sat}$"),
    offsets=[1e18, None, None, None],
    fig_height_per_row=1.36,
    extra_args=(
        dict(),
        dict(),
        dict(diverging=True, cmap="RdBu_r", cbar_lim_=[-1.2, 1.2]),
        dict(log_cbar=False, cbar_lim_=[0, 1.0]),
    ),
    subplots_kwargs=dict(hspace=0.05, wspace=0.05),
    n_contours=11,
    show=show_figures,
)

# %%
tile2d(
    experimental_data,
    simulation_data,
    ("RDPA", "RDPA", "RDPA", "RDPA"),
    ("electron_temp", "potential", "vfloat", "vfloat_std"),
    labels=("electron temp", "potential", "$V_{fl}$", "Std. dev. of $V_{fl}$"),
    fig_height_per_row=1.36,
    subplots_kwargs=dict(hspace=0.05, wspace=0.05),
    n_contours=11,
    show=show_figures,
)
