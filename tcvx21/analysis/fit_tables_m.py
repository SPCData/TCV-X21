"""
Makes tables for fitted profiles
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from collections import namedtuple
import tcvx21
from tcvx21.analysis import fit_exponential_decay, fit_eich_profile
from tcvx21.analysis.fit_eich_profile_m import fit_eich_profile, eich_profile
from tcvx21.plotting.labels_m import label_subplots

from tcvx21.plotting.labels_m import (
    add_twinx_label,
    make_labels,
    add_x_zero_line,
    add_y_zero_line,
    format_yaxis,
)
from tcvx21.quant_validation.ricci_metric_m import (
    calculate_normalised_ricci_distance_from_observables,
)
from tcvx21.plotting import savefig


def decay_rate_row(
    experimental_data,
    simulation_data,
    field_direction,
    diagnostic,
    observable,
    row_label,
    fit_range=(0, 1.0),
    plot=False,
):
    """
    Make rows of the decay rate table (and plot if requested)
    """
    Row = namedtuple("Row", ("TCV", *simulation_data.keys()))
    results = []

    fd = "+" if field_direction == "forward_field" else "-"
    if plot:
        plt.figure()

    ref = experimental_data[field_direction].get_observable(diagnostic, observable)
    _, (K, K_err) = fit_exponential_decay(
        ref.positions,
        ref.values,
        ref.errors,
        fit_range,
        plot=f"TCV{fd}" if plot else "",
    )
    results.append(
        f"{K.to('cm').magnitude:.1f}$\\pm${K_err.to('cm').magnitude:.1f}"
        if K is not None
        else "-"
    )

    for key, case in simulation_data.items():
        sim = case[field_direction].get_observable(diagnostic, observable)
        if sim.is_empty:
            K = None
        else:
            _, (K, K_err) = fit_exponential_decay(
                sim.positions,
                sim.values,
                sim.errors if sim.has_errors else None,
                fit_range,
                plot=f"{key}{fd}" if plot else "",
            )
        results.append(
            f"{K.to('cm').magnitude:.1f}$\\pm${K_err.to('cm').magnitude:.1f}"
            if K is not None
            else "-"
        )

    if plot:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(f"{row_label}$_{{{fd}}}$")

    return f"$\\lambda_{{{row_label}}}$$^{{{fd}}}$", Row(*results)


def make_decay_rate_table(
    experimental_data,
    simulation_data,
    diagnostics,
    observables,
    labels,
    fit_range=(0, 1.0),
    plot=False,
    output_path: Path = None,
):
    """
    Makes a table giving the exponential decay rates near the separatrix for a given list of diagnostics and observables
    Writes a LaTeX table as output
    """
    assert (
        np.array(diagnostics).size
        == np.array(observables).size
        == np.array(labels).size
    )

    rows, extended_labels = [], []
    for field_direction in ["forward_field", "reversed_field"]:
        for diagnostic, observable, label in zip(diagnostics, observables, labels):

            label, row = decay_rate_row(
                experimental_data,
                simulation_data,
                field_direction,
                diagnostic,
                observable,
                label,
                fit_range=fit_range,
                plot=plot,
            )

            rows.append(row)
            extended_labels.append(label)

    df = pd.DataFrame(rows, index=extended_labels)
    if output_path is None:
        output_path = tcvx21.results_dir / "tables" / "exp_decay.tex"
    df.to_latex(
        output_path,
        float_format="{:0.1f}".format,
        escape=False,
        column_format=f"l{(len(simulation_data)+1)*'c'}",
    )

    return df


def style_eich_plot(ax, reference):
    add_x_zero_line(ax)
    add_y_zero_line(ax)
    ax.set_xlim(reference.xlim)
    reference.apply_plot_limits(ax)
    ax.set_xticks([-1, 0, 1, 2])
    plt.legend()


def append_to_list_as_str(list_in, values=None, units=None):
    if values is None:
        list_in.append("-")
    else:
        values = values.to(units).magnitude
        list_in.append(f"{values[0]:.1f}$\\pm${values[1]:.1f}")


def eich_comparison(
    experimental_data, simulation_data, field_direction, diagnostic, observable, ax
):
    lambda_q_list, S_list = [], []

    reference = experimental_data[field_direction].get_observable(
        diagnostic, observable
    )
    reference.plot(ax)

    peak_heat_flux, lambda_q, spreading, background_heat_flux, shift = fit_eich_profile(
        reference.positions, reference.values
    )
    append_to_list_as_str(lambda_q_list, lambda_q, "mm")
    append_to_list_as_str(S_list, spreading, "mm")

    ax.plot(
        reference.positions,
        eich_profile(
            reference.positions,
            peak_heat_flux[0],
            lambda_q[0],
            spreading[0],
            background_heat_flux[0],
            shift[0],
        ),
        linestyle="--",
        linewidth=0.75 * plt.rcParams["lines.linewidth"],
    )

    field_direction_string, diagnostic_string, observable_string = make_labels(
        field_direction, diagnostic, reference
    )
    add_twinx_label(ax, field_direction_string)
    ax.set_title(f"{diagnostic_string}: {observable_string}")

    for key, case in simulation_data.items():

        simulation = case[field_direction].get_observable(diagnostic, observable)
        if simulation.is_empty:
            append_to_list_as_str(lambda_q_list, None)
            append_to_list_as_str(S_list, None)
            continue
        line = simulation.plot(ax, trim_to_x=reference.xlim, linestyle="-")

        (
            peak_heat_flux,
            lambda_q,
            spreading,
            background_heat_flux,
            shift,
        ) = fit_eich_profile(simulation.positions, simulation.values)
        append_to_list_as_str(lambda_q_list, lambda_q, "mm")
        append_to_list_as_str(S_list, spreading, "mm")

        ax.plot(
            simulation.positions,
            eich_profile(
                simulation.positions,
                peak_heat_flux[0],
                lambda_q[0],
                spreading[0],
                background_heat_flux[0],
                shift[0],
            ),
            linestyle="--",
            linewidth=0.75 * plt.rcParams["lines.linewidth"],
            color=simulation.color,
        )

        d = calculate_normalised_ricci_distance_from_observables(
            simulation=simulation, experiment=reference
        )
        line.set_label(f"{line.get_label()}: d={d:3.2f}")

    style_eich_plot(ax, reference)

    return lambda_q_list, S_list


def make_eich_fit_comparison(
    experimental_data,
    simulation_data,
    diagnostic="LFS-IR",
    observable="q_parallel",
    table_output_path: Path = None,
    fig_output_path: Path = None,
):

    Row = namedtuple("Row", ("TCV", *simulation_data.keys()))
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.25, 5))

    rows, labels = [], []
    for field_direction, ax in zip(["forward_field", "reversed_field"], axs.flatten()):
        fd = "+" if field_direction == "forward_field" else "-"
        labels.append(f"$\\lambda_{{q}}^{{{fd}}}$")
        labels.append(f"$S^{{{fd}}}$")

        (lambda_q_list, S_list) = eich_comparison(
            experimental_data,
            simulation_data,
            field_direction,
            diagnostic,
            observable,
            ax,
        )

        rows.append(Row(*lambda_q_list))
        rows.append(Row(*S_list))

    axs[0].set_xlabel("")
    axs[1].set_title("")
    axs[0].set_ylim(bottom=0)
    axs[1].set_ylim(bottom=0)
    label_subplots(axs)

    plt.draw()
    for ax in axs.flatten():
        format_yaxis(ax)

    if fig_output_path is None:
        fig_output_path = tcvx21.results_dir / "analysis_fig" / "eich_fit.png"
    savefig(fig, fig_output_path)

    df = pd.DataFrame(rows, index=labels)
    if table_output_path is None:
        table_output_path = tcvx21.results_dir / "tables" / "lambda_q.tex"
    df.to_latex(
        table_output_path,
        float_format="{:0.1f}".format,
        escape=False,
        column_format=f"l{(len(simulation_data)+1)*'c'}",
    )

    return df
