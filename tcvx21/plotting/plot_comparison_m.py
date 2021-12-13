"""
Routines for making plots to compare results for single observables
"""
import matplotlib.pyplot as plt
import tcvx21

plt.style.use(tcvx21.style_sheet)

from .save_figure_m import savefig
from .plot_comparison_1d_m import plot_1D_comparison
from .plot_comparison_2d_m import plot_2D_comparison, plot_2D_comparison_simulation_only


def plot_comparison(
    field_direction: str,
    diagnostic: str,
    observable: str,
    experimental_data: dict,
    simulation_data: dict,
    show: bool = False,
    save: bool = True,
    close: bool = True,
    output_path=None,
    debug: bool = False,
    **kwargs,
):
    """
    Plots a comparison of experimental_data against simulation_data for a single
    field direction, diagnostic and observable

    simulation_data should be dictionary of the form
    {'simulation_name': {'forward_field': Record(), 'reversed_field': Record}}
    """
    reference = experimental_data[field_direction].get_observable(
        diagnostic, observable
    )

    if reference.is_empty:
        if debug:
            print(
                f"Missing experimental data for {field_direction}:{diagnostic}:{observable}"
            )
        reference = list(simulation_data.values())[0][field_direction].get_observable(
            diagnostic, observable
        )
        simulation_only = True
    else:
        simulation_only = False

    dimensionality = reference.dimensionality

    keys = dict(
        field_direction=field_direction, diagnostic=diagnostic, observable=observable
    )
    kwargs = {**kwargs, **keys}

    if dimensionality == 1:
        fig, ax = plot_1D_comparison(
            **kwargs,
            experimental_data=experimental_data,
            simulation_data=simulation_data,
        )

    else:
        if simulation_only:
            fig, ax = plot_2D_comparison_simulation_only(
                **kwargs, simulation_data=simulation_data
            )

        else:
            fig, ax = plot_2D_comparison(
                **kwargs,
                experimental_data=experimental_data,
                simulation_data=simulation_data,
            )

    if output_path is None and save:
        output_path = (
            tcvx21.results_dir
            / "observables_fig"
            / diagnostic
            / f"{diagnostic}_{observable}_{field_direction}.png"
        )

    savefig(fig, output_path=output_path, show=show, close=close)

    return fig, ax
