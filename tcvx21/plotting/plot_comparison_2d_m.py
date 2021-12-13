"""
Plotting routines for 2D observables
"""
import matplotlib.pyplot as plt
import tcvx21

plt.style.use(tcvx21.style_sheet)

from tcvx21.quant_validation.ricci_metric_m import (
    calculate_normalised_ricci_distance_from_observables,
)
from .labels_m import add_twinx_label, make_labels, add_x_zero_line, make_colorbar
import numpy as np


def plot_2D_comparison(
    field_direction,
    diagnostic,
    observable,
    experimental_data,
    simulation_data,
    fig_width=7.5,
    fig_height_per_row=1.5,
    title_height=1.0,
    common_colorbar: bool = True,
    experiment_sets_colorbar: bool = False,
    diverging: bool = False,
    **kwargs,
):
    """
    Plots a 2D (area data) observable, with different values represented via a constant color

    common_colorbar will use the same colorbar for all figures, which is good for a quantitative side-by-side
    comparison, but which is sensitive to extreme values
    """
    fig, axs = plt.subplots(
        ncols=len(simulation_data) + 1,
        figsize=(fig_width, title_height + fig_height_per_row),
        sharex=True,
        sharey=True,
    )

    for ax in axs.flatten():
        add_x_zero_line(ax)

    key = {"diagnostic": diagnostic, "observable": observable}

    reference = experimental_data[field_direction].get_observable(**key)

    simulations = [
        case[field_direction].get_observable(**key) for case in simulation_data.values()
    ]
    if common_colorbar:
        if experiment_sets_colorbar:
            cbar_lim = reference.calculate_cbar_limits().to(reference.compact_units)
        else:
            cbar_lim = reference.calculate_cbar_limits(simulations).to(
                reference.compact_units
            )

        if diverging:
            cbar_lim[0], cbar_lim[1] = -max(np.abs(cbar_lim)), max(np.abs(cbar_lim))
    else:
        cbar_lim = None

    image = reference.plot(axs[0], cbar_lim=cbar_lim, diverging=diverging, **kwargs)
    title_text = axs[0].title.get_text()
    axs[0].title.set_text("")
    axs[0].set_title(f"{title_text}", loc="left", fontsize="small")
    axs[0].set_ylim(reference.positions_zx.min(), reference.positions_zx.max())

    for simulation, ax in zip(simulations, axs[1:]):

        if simulation.is_empty:
            continue
        image = simulation.plot(
            ax,
            units=reference.compact_units,
            cbar_lim=cbar_lim,
            diverging=diverging,
            **kwargs,
        )
        ax.set_ylabel("")

        d = calculate_normalised_ricci_distance_from_observables(
            simulation=simulation, experiment=reference
        )

        title_text = ax.title.get_text()
        ax.title.set_text("")
        ax.set_title(f"{title_text}: d={d:2.1f}", loc="left", fontsize="small")

    _, fig_height = fig.get_size_inches()
    plt.subplots_adjust(top=1 - title_height / fig_height, wspace=0.25)

    field_direction_string, _, observable_string = make_labels(
        field_direction, diagnostic, reference
    )

    if common_colorbar:
        plt.suptitle(
            f"{observable_string}: {field_direction_string}",
            fontsize="large",
            y=1 - title_height / 2 / fig_height,
        )
        cbar = make_colorbar(
            ax=axs.flatten(),
            mappable=image,
            units=reference.compact_units,
            as_title=True,
        )
        ticks = np.round(
            np.linspace(cbar_lim[0].magnitude, cbar_lim[1].magnitude, num=5), decimals=1
        )
        tick_labels = [f"{x:4.3}" for x in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    else:
        plt.suptitle(
            observable_string, fontsize="large", y=1 - title_height / 2 / fig_height
        )
        add_twinx_label(axs[-1], field_direction_string)

    return fig, axs


def plot_2D_comparison_simulation_only(
    field_direction,
    diagnostic,
    observable,
    simulation_data,
    fig_width=7.5,
    fig_height_per_row=1.5,
    title_height=1.0,
    common_colorbar: bool = True,
    experiment_sets_colorbar: bool = False,
    diverging: bool = False,
    **kwargs,
):
    """
    Plots a 2D (area data) observable, with different values represented via a constant color

    common_colorbar will use the same colorbar for all figures, which is good for a quantitative side-by-side
    comparison, but which is sensitive to extreme values
    """
    fig, axs = plt.subplots(
        ncols=len(simulation_data),
        figsize=(fig_width, title_height + fig_height_per_row),
        sharex=True,
        sharey=True,
    )
    axs = np.atleast_1d(axs)

    for ax in axs.flatten():
        add_x_zero_line(ax)

    key = {"diagnostic": diagnostic, "observable": observable}

    reference = list(simulation_data.values())[0][field_direction].get_observable(**key)

    simulations = [
        case[field_direction].get_observable(**key) for case in simulation_data.values()
    ]
    if common_colorbar:
        if experiment_sets_colorbar:
            print(
                "Warning: experiment_sets_colorbar ignored in plot_2D_comparison_simulation_only"
            )
        else:
            cbar_lim = reference.calculate_cbar_limits(simulations).to(
                reference.compact_units
            )

        if diverging:
            cbar_lim[0], cbar_lim[1] = -max(np.abs(cbar_lim)), max(np.abs(cbar_lim))
    else:
        cbar_lim = None

    axs[0].set_ylim(reference.positions_zx.min(), reference.positions_zx.max())

    for simulation, ax in zip(simulations, axs):

        if simulation.is_empty:
            continue
        image = simulation.plot(
            ax,
            units=reference.compact_units,
            cbar_lim=cbar_lim,
            diverging=diverging,
            **kwargs,
        )
        ax.set_ylabel("")

        title_text = ax.title.get_text()
        ax.title.set_text("")
        ax.set_title(f"{title_text}", loc="left", fontsize="small")

    _, fig_height = fig.get_size_inches()
    plt.subplots_adjust(top=1 - title_height / fig_height, wspace=0.25)

    field_direction_string, _, observable_string = make_labels(
        field_direction, diagnostic, reference
    )

    if common_colorbar:
        plt.suptitle(
            f"{observable_string}: {field_direction_string}",
            fontsize="large",
            y=1 - title_height / 2 / fig_height,
        )
        cbar = make_colorbar(
            ax=axs.flatten(),
            mappable=image,
            units=reference.compact_units,
            as_title=True,
        )
        ticks = np.round(
            np.linspace(cbar_lim[0].magnitude, cbar_lim[1].magnitude, num=5), decimals=1
        )
        tick_labels = [f"{x:4.3}" for x in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    else:
        plt.suptitle(
            observable_string, fontsize="large", y=1 - title_height / 2 / fig_height
        )
        add_twinx_label(axs[-1], field_direction_string)

    return fig, axs
