"""
Plotting routines for 1D observables
"""
import matplotlib.pyplot as plt
import tcvx21

plt.style.use(tcvx21.style_sheet)

from tcvx21.quant_validation.ricci_metric_m import (
    calculate_normalised_ricci_distance_from_observables,
)
from tcvx21.plotting.labels_m import (
    add_twinx_label,
    add_x_zero_line,
    add_y_zero_line,
    make_labels,
    format_yaxis,
)

from tcvx21.quant_validation.latex_table_writer_m import observable_latex


def plot_1D_comparison(
    field_direction,
    diagnostic,
    observable,
    experimental_data,
    simulation_data,
    **kwargs,
):
    """
    Plots a 1D (line data) observable, with different values represented via a constant color
    """
    fig, ax = plt.subplots()

    add_x_zero_line(ax)
    add_y_zero_line(ax)

    if "kurtosis" in observable:
        add_y_zero_line(ax, level=3.0)

    reference = experimental_data[field_direction].get_observable(
        diagnostic, observable
    )
    reference_available = not (reference.is_empty)
    if reference_available:
        reference.plot(ax, **kwargs)
    else:
        # Use one of the simulations as the "reference" to access the properties
        reference = list(simulation_data.values())[0][field_direction].get_observable(
            diagnostic, observable
        )

    field_direction_string, diagnostic_string, observable_string = make_labels(
        field_direction, diagnostic, reference
    )

    add_twinx_label(ax, field_direction_string)

    ax.set_title(f"{diagnostic_string}: {observable_string}")

    for case in simulation_data.values():

        simulation = case[field_direction].get_observable(diagnostic, observable)
        if simulation.is_empty:
            continue

        line = simulation.plot(ax, trim_to_x=reference.xlim, **kwargs)
        if reference_available:
            d = calculate_normalised_ricci_distance_from_observables(
                simulation=simulation, experiment=reference
            )
            line.set_label(f"{line.get_label()}: d={d:3.2f}")
        else:
            line.set_label(f"{line.get_label()}")

    # Set the ylim to show all data
    ax.set_ylim(
        *reference.ylims_in_trim(
            [
                case[field_direction].get_observable(diagnostic, observable)
                for case in simulation_data.values()
                if not case[field_direction].is_empty
            ],
            trim_to_x=reference.xlim,
        )
    )

    # Apply custom limits if they are defined
    reference.apply_plot_limits(ax)
    plt.draw()
    format_yaxis(ax)

    plt.legend()

    return fig, ax
