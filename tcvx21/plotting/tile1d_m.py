"""
Routine for making a tiled plot

This is helpful for comparing results with each other

Can make tile plots either for a single diagnostic and many observables, or for a single observable at many
diagnostic positions
"""
import matplotlib.pyplot as plt
import tcvx21

plt.style.use(tcvx21.style_sheet)

import numpy as np
from pathlib import Path

from tcvx21.plotting.labels_m import (
    add_twinx_label,
    add_x_zero_line,
    add_y_zero_line,
    make_observable_string,
    make_diagnostic_string,
    make_field_direction_string,
    label_subplots,
    format_yaxis,
)
from tcvx21.quant_validation.ricci_metric_m import (
    calculate_normalised_ricci_distance_from_observables,
)
from .save_figure_m import savefig

from typing import Union
from tcvx21.quant_validation.latex_table_writer_m import observable_latex


def tile1d(
    experimental_data,
    simulation_data,
    diagnostics: Union[np.ndarray, str, tuple],
    observables: Union[np.ndarray, str, tuple],
    overplot: tuple = None,
    fig_width=7.5,
    fig_height_per_row=2.0,
    title_height=1.0,
    manual_title: str = "",
    legend_loc=("best",),
    make_title: bool = True,
    show: bool = False,
    save: bool = True,
    close: bool = True,
    output_path: Path = None,
):
    """
    Make a tiled plot showing either several observables for a single diagnostics, or several diagnostics (positions)
    for the same observable
    """
    observables = np.atleast_1d(observables)
    diagnostics = np.atleast_1d(diagnostics)

    assert observables.ndim == 1 and diagnostics.ndim == 1

    if observables.size == 1:
        mode = "observable"
        observables = np.broadcast_to(observables, diagnostics.size)
    elif diagnostics.size == 1:
        mode = "diagnostic"
        diagnostics = np.broadcast_to(diagnostics, observables.size)
    else:
        raise NotImplementedError(
            f"Could not make tile plot with observables {observables.shape} "
            f"and diagnostics {diagnostics.shape}"
        )

    ncols = 2
    nrows = observables.size

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(fig_width, title_height + nrows * fig_height_per_row),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    for column, field_direction in enumerate(["forward_field", "reversed_field"]):
        # Always plot forward field in the left column, and reversed field in the
        # right column
        for row, (observable, diagnostic) in enumerate(zip(observables, diagnostics)):
            cases = []
            for fd in ["forward_field", "reversed_field"]:
                c = experimental_data[fd].get_observable(diagnostic, observable)
                if not c.is_empty:
                    cases.append(c)

            ax = axs[row, column]

            reference = experimental_data[field_direction].get_observable(
                diagnostic, observable
            )
            if reference.is_empty:
                continue

            if mode == "diagnostic":
                label = diagnostic
            else:
                label = f"TCV {observable_latex[observable]}"

            reference.plot(ax, label=label)

            for case in simulation_data.values():
                for fd in ["forward_field", "reversed_field"]:
                    c = case[fd].get_observable(diagnostic, observable)
                    if not c.is_empty:
                        cases.append(c)

                simulation = case[field_direction].get_observable(
                    diagnostic, observable
                )
                if simulation.is_empty:
                    continue

                line = simulation.plot(ax, trim_to_x=reference.xlim)

                d = calculate_normalised_ricci_distance_from_observables(
                    simulation=simulation, experiment=reference
                )

                line.set_label(f"{line.get_label()}: d={d:3.2f}")

            if overplot is not None:
                if mode == "observable":
                    raise NotImplementedError(
                        "Overplotting of observables is not supported (units must match)"
                    )
                for (
                    overplot_observable,
                    overplot_diagnostic,
                    overplot_color,
                ) in overplot:
                    if mode == "diagnostic" and observable == overplot_observable:
                        overplot_ = experimental_data[field_direction].get_observable(
                            overplot_diagnostic, observable
                        )
                        if overplot_.is_empty:
                            continue
                        overplot_.plot(
                            ax, color=overplot_color, label=overplot_diagnostic
                        )

            # Set the ylim to show all data
            ax.set_ylim(*reference.ylims_in_trim(cases, trim_to_x=reference.xlim))

            # Apply custom limits if they are defined
            reference.apply_plot_limits(ax)

            ax.legend(
                loc=legend_loc[0]
                if len(legend_loc) == 1
                else legend_loc[2 * row + column]
            )
            add_x_zero_line(ax)
            add_y_zero_line(ax)
            if "kurtosis" in observable:
                add_y_zero_line(ax, level=3.0)

            if row != nrows - 1:
                ax.set_xlabel("")
            if column != 0:
                ax.set_ylabel("")
                if mode == "observable":
                    add_twinx_label(ax, make_diagnostic_string(diagnostic))
                else:
                    add_twinx_label(ax, make_observable_string(reference))

            if row == 0:
                ax.set_title(make_field_direction_string(field_direction))

    plt.draw()
    for row in range(nrows):
        format_yaxis(axs[row, 0])

    if manual_title:
        title_string = manual_title
    elif mode == "observable":
        reference = experimental_data["forward_field"].get_observable(
            diagnostics[0], observables[0]
        )
        title_string = make_observable_string(reference)
    else:
        title_string = make_diagnostic_string(diagnostics[0])

    if make_title:
        _, fig_height = fig.get_size_inches()
        plt.subplots_adjust(top=1 - title_height / fig_height)

        plt.suptitle(
            title_string, fontsize="large", y=1 - title_height / 2 / fig_height
        )

    label_subplots(axs.flatten())

    if output_path is None and save:
        if mode == "observable":
            filename = f"{title_string}+{','.join(list(diagnostics))}".replace(" ", "_")
        else:
            filename = f"{title_string}+{','.join(list(observables))}".replace(" ", "_")

        output_path = tcvx21.results_dir / "summary_fig" / f"{filename}.png"

    savefig(fig, output_path=output_path, show=show, close=close)

    return fig, axs
