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
from typing import Union
from .tile2d_single_observable import plot_2D_observables


def tile2d(
    experimental_data,
    simulation_data,
    diagnostics: Union[np.ndarray, str, tuple],
    observables: Union[np.ndarray, str, tuple],
    labels: list,
    fig_width=7.5,
    fig_height_per_row=2.0,
    title_height=1.0,
    manual_title: str = "",
    extra_args=None,
    x_title=0.875,
    subplots_kwargs={},
    offsets: np.ndarray = None,
    show: bool = False,
    save: bool = True,
    close: bool = True,
    output_path: Path = None,
    **kwargs,
):
    """
    Plots multiple 2D observables
    extra_args should be a tuple or list of the length of the diagnostics and observables, which contains
    keyword arguments passed to each key-set
    """
    assert len(diagnostics) == len(observables)
    assert len(labels) == len(observables)
    if extra_args is not None:
        assert len(extra_args) == len(observables)

    nrows = 2 * len(observables)

    fig, axs = plt.subplots(
        ncols=len(simulation_data) + 1,
        nrows=nrows,
        figsize=(fig_width, nrows * fig_height_per_row),
        sharex="col",
        sharey="row",
    )

    if subplots_kwargs:
        plt.subplots_adjust(**subplots_kwargs)

    if offsets is None:
        offsets = len(observables) * [None]

    for index, key in enumerate(zip(diagnostics, observables)):

        if extra_args is not None:
            key_kwargs = extra_args[index]
        else:
            key_kwargs = {}

        subaxs = axs[2 * index : 2 * index + 2]
        label_subplots(subaxs, prefix=f"{index+1}")

        _, cbar = plot_2D_observables(
            axs=subaxs,
            key=key,
            experimental_data=experimental_data,
            simulation_data=simulation_data,
            offset=offsets[index],
            cbar_pad=0.025,
            add_labels=False,
            **key_kwargs,
            **kwargs,
        )

        reference = experimental_data["forward_field"].get_observable(*key)
        compact_units = reference.compact_units
        units_string = (
            f"{Quantity(1, compact_units).units:~P}" if compact_units else "-"
        )
        if offsets[index] is not None:
            units_string = (
                f"$10^{{{np.log10(offsets[index]):n}}}${units_string.lstrip('1')}"
            )

        fig.text(
            x_title,
            (subaxs[-1, 0].get_position().y1 + subaxs[-1, 1].get_position().y1) / 2,
            f"{labels[index]} [{units_string}]",
            va="center",
            rotation=270,
        )

    for ax, title in zip(axs[0], ["TCV", *simulation_data.keys()]):
        ax.set_title(title)

    fig.text(0.5, 0.09, "$R^u - R^u_{sep}$ [cm]", ha="center")
    fig.text(0.02, 0.5, "$Z - Z_X$ [m]", va="center", rotation="vertical")

    if output_path is None and save:

        filename = "+".join(
            [
                f"{diagnostic}_{observable}"
                for diagnostic, observable in zip(diagnostics, observables)
            ]
        )

        output_path = (
            tcvx21.results_dir / "summary_fig" / f"{filename.replace(' ', '_')}.png"
        )

    savefig(fig, output_path=output_path, show=show, close=close)

    return fig, axs
