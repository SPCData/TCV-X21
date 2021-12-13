import matplotlib.pyplot as plt
import tcvx21

plt.style.use(tcvx21.style_sheet)

from pathlib import Path
from tcvx21 import Quantity
from tcvx21.quant_validation.ricci_metric_m import (
    calculate_normalised_ricci_distance_from_observables,
)
from tcvx21.plotting.labels_m import (
    add_twinx_label,
    add_x_zero_line,
    make_observable_string,
    label_subplots,
)
import numpy as np
from .save_figure_m import savefig


def tile2d_single_observable(
    experimental_data,
    simulation_data,
    diagnostic,
    observable,
    fig_width=7.5,
    fig_height_per_row=1.5,
    title_height=1.0,
    subplots_kwargs={},
    offset=None,
    save: bool = True,
    show: bool = False,
    close: bool = True,
    output_path: Path = None,
    **kwargs,
):
    """
    Plots both field directions of a single 2d observable
    """
    nrows = 2
    key = (diagnostic, observable)

    fig, axs = plt.subplots(
        ncols=len(simulation_data) + 1,
        nrows=nrows,
        figsize=(fig_width, title_height + nrows * fig_height_per_row),
        sharex="col",
        sharey="row",
    )

    plt.subplots_adjust(
        top=1 - title_height / fig.get_size_inches()[1], **subplots_kwargs
    )

    plot_2D_observables(
        axs=axs,
        key=key,
        experimental_data=experimental_data,
        simulation_data=simulation_data,
        offset=offset,
        **kwargs,
    )

    reference = experimental_data["forward_field"].get_observable(*key)
    compact_units = reference.compact_units
    units_string = f"{Quantity(1, compact_units).units:~P}" if compact_units else "-"
    if offset is not None:
        units_string = f"$10^{{{np.log10(offset):n}}}${units_string.lstrip('1')}"
    plt.suptitle(
        f"{make_observable_string(reference)} [{units_string}]",
        fontsize="large",
        y=1 - title_height / 2 / fig.get_size_inches()[1],
    )

    label_subplots(axs.flatten())

    if output_path is None and save:
        output_path = (
            tcvx21.results_dir / "summary_fig" / f"{diagnostic}+{observable}.png"
        )

    savefig(fig, output_path=output_path, show=show, close=close)

    return fig, axs


def plot_2D_observables(
    axs,
    key,
    experimental_data,
    simulation_data,
    cbar_lim_=(None, None),
    experiment_sets_cbar=True,
    add_labels=True,
    cbar_pad=0.075,
    diverging: bool = False,
    log_cbar: bool = False,
    offset=None,
    ticks=None,
    **kwargs,
):
    """
    Routine to plot and style a 2D observable

    key = (diagnostic, observable)
    """
    # Load the experimental data
    reference = {
        "forward_field": experimental_data["forward_field"].get_observable(*key),
        "reversed_field": experimental_data["reversed_field"].get_observable(*key),
    }

    # Calculate the common colorbar
    if experiment_sets_cbar:
        cbar_lim = reference["forward_field"].calculate_cbar_limits(
            (reference["reversed_field"],)
        )
    else:
        cbar_lim = reference["forward_field"].calculate_cbar_limits(
            [
                simulation["forward_field"].get_observable(*key)
                for simulation in simulation_data.values()
            ]
            + [
                simulation["reversed_field"].get_observable(*key)
                for simulation in simulation_data.values()
            ]
        )

    cbar_lim[0] = cbar_lim[0] if cbar_lim_[0] is None else cbar_lim_[0]
    cbar_lim[1] = cbar_lim[1] if cbar_lim_[1] is None else cbar_lim_[1]
    if diverging:
        cbar_lim[0], cbar_lim[1] = -max(np.abs(cbar_lim)), max(np.abs(cbar_lim))
    cbar_lim = cbar_lim.to(reference["forward_field"].compact_units)

    # Plot the experimental data
    image = reference["forward_field"].plot(
        axs[0][0], cbar_lim=cbar_lim, log_cbar=log_cbar, **kwargs
    )
    image = reference["reversed_field"].plot(
        axs[1][0], cbar_lim=cbar_lim, log_cbar=log_cbar, **kwargs
    )

    for row, field_direction in enumerate(["forward_field", "reversed_field"]):
        # Set labels for the experimental data
        axs[row][0].set_ylim(
            reference[field_direction].positions_zx.min(),
            reference[field_direction].positions_zx.max(),
        )
        if add_labels:
            title_text = axs[row][0].title.get_text()
            axs[row][0].title.set_text("")
            axs[row][0].set_title(f"{title_text}", fontsize="small", pad=2.0)

        for simulation, ax in zip(
            [
                case[field_direction].get_observable(*key)
                for case in simulation_data.values()
            ],
            axs[row, 1:],
        ):
            # Plot the simulation data
            if simulation.is_empty:
                ax.set_axis_off()
                ax.set_visible(False)
                continue

            simulation.plot(
                ax,
                units=reference[field_direction].compact_units,
                cbar_lim=cbar_lim,
                log_cbar=log_cbar,
                **kwargs,
            )
            ax.set_ylabel("")
            add_x_zero_line(ax)

            if add_labels:
                d = calculate_normalised_ricci_distance_from_observables(
                    simulation=simulation, experiment=reference[field_direction]
                )

                title_text = ax.title.get_text()
                ax.title.set_text("")
                ax.set_title(f"{title_text}: d={d:2.1f}", fontsize="small", pad=2.0)

    if not add_labels:
        for ax in axs.flatten():
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
        for field_direction, ax in zip(["Forward", "Reversed"], axs[:, 0].flatten()):
            ax.set_ylabel(field_direction)
    else:
        for ax in axs[0, :].flatten():
            ax.set_xlabel("")
        for field_direction, ax in zip(
            ["Forward field", "Reversed field"], axs[:, -1].flatten()
        ):
            add_twinx_label(ax, field_direction, labelpad=15, visible=ax.get_visible())

    if ticks is None:
        if log_cbar:
            ticks = np.logspace(
                np.log10(cbar_lim[0].magnitude), np.log10(cbar_lim[1].magnitude), num=5
            )
        else:
            ticks = np.linspace(cbar_lim[0].magnitude, cbar_lim[1].magnitude, num=5)

    if log_cbar:

        def format_func(value, tick_number):
            return f"$10^{{{np.log10(value):2.1f}}}$"

    else:

        def format_func(value, tick_number):
            offset_ = offset if offset is not None else 1.0
            return f"{value/ offset_:.3g}"

    cbar = plt.colorbar(
        image,
        ax=axs.flatten(),
        pad=cbar_pad,
        ticks=ticks,
        format=plt.FuncFormatter(format_func),
    )

    return axs, cbar
