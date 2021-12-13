"""
Routines for calculating the Ricci validation metric
"""
import warnings

import numpy as np
import pandas as pd
from collections import namedtuple

from tcvx21 import Quantity
from tcvx21.record_c import Record
from tcvx21.observable_c import Observable


def compute_comparison_data(experiment: Observable, simulation: Observable):
    """
    Calculates the terms to compute the metric terms
    """

    if experiment.dimensionality == 0 and simulation.dimensionality == 0:
        mapped_simulation = simulation
        mask = [True]

    elif experiment.dimensionality == 1 and simulation.dimensionality == 1:
        mask = simulation.points_overlap(experiment.positions)
        mapped_simulation = simulation.interpolate_onto_positions(
            experiment.positions[mask]
        )

    elif experiment.dimensionality == 2 and simulation.dimensionality == 2:
        mapped_simulation = simulation.interpolate_onto_reference(experiment)
        mask = mapped_simulation.mask

    else:
        raise NotImplementedError(
            f"No implementation for comparing {type(experiment)} and {type(simulation)}"
        )

    result = {
        "experimental_value": experiment.values[mask],
        "experimental_error": experiment.errors[mask],
        "simulation_value": mapped_simulation.values,
        "simulation_error": mapped_simulation.errors,
    }

    return result


def calculate_normalised_ricci_distance_from_observables(
    experiment: Observable, simulation: Observable
):
    """
    Computes the d value between an experiment and a simulation
    """

    return calculate_normalised_ricci_distance(
        **compute_comparison_data(experiment, simulation)
    )


def calculate_normalised_ricci_distance(
    experimental_value: Quantity,
    experimental_error: Quantity,
    simulation_value: Quantity,
    simulation_error: Quantity,
    debug: bool = False,
):
    """
    Calculates the error-normalised Ricci distance between an experimental signal and a simulation.

    Note that the simulation and experiment values must be arrays at the same position. To achieve this,
    call interpolate_onto_experimental_points and pass the returned mapped simulation values, plus the experimental
    values with the returned mask.
    """
    N_points = experimental_value.size
    assert np.all(
        np.array(
            [experimental_error.size, simulation_value.size, simulation_value.size]
        )
        == N_points
    ), f"{[N_points, experimental_error.size, simulation_value.size, simulation_value.size]}"

    return (
        np.sqrt(
            1.0
            / N_points
            * np.sum(
                (experimental_value - simulation_value) ** 2
                / (experimental_error ** 2 + simulation_error ** 2),
            )
        )
        .to("")
        .magnitude
    )


def calculate_ricci_sensitivity(
    experimental_value: Quantity,
    experimental_error: Quantity,
    simulation_value: Quantity,
    simulation_error: Quantity,
    float=1e-3,
):
    """
    Calculates the Ricci sensitivity, which is a measure of the total uncertainty relative to the magnitude of
    the values.

    Note that the simulation and experiment values must be arrays at the same position. To achieve this,
    call interpolate_onto_experimental_points and pass the returned mapped simulation values, plus the experimental
    values with the returned mask.
    """

    return (
        np.exp(
            -1
            * (np.sum(np.abs(experimental_error) + np.abs(simulation_error)))
            / (np.sum(np.abs(experimental_value) + np.abs(simulation_value)))
        )
        .to("")
        .magnitude
    )


def level_of_agreement_function(
    distance: float, agreement_threshold=1.0, transition_sharpness=0.5
) -> float:
    """
    Calculates the level of agreement (R) from the normalised distance (d)
    """
    return (
        np.tanh(
            (distance - 1.0 / distance - agreement_threshold) / transition_sharpness
        )
        + 1.0
    ) / 2.0


class RicciValidation:
    def __init__(self, experimental_dataset: Record, simulation_dataset: Record):
        """
        Stores a single experiment and a single experiment, to allow easy iteration over all stored elements
        """
        self.experiment = experimental_dataset
        self.simulation = simulation_dataset

        # Skip observables where there is no experimental data
        self.keys = [
            key
            for key in self.experiment.keys()
            if not experimental_dataset.get_observable(*key).is_empty
        ]

        self.distance, self.hierarchy, self.sensitivity = {}, {}, {}

        for dictionary in [self.distance, self.sensitivity]:
            for diagnostic, observable in self.keys:

                dictionary.setdefault(diagnostic, {})
                dictionary[diagnostic].setdefault(observable, np.nan)

        for diagnostic, observable in self.keys:

            self.hierarchy.setdefault(diagnostic, {})

            ds_entry = self.simulation.get_nc_group(diagnostic, observable)

            self.hierarchy[diagnostic][observable] = (
                ds_entry.experimental_hierarchy + ds_entry.simulation_hierarchy - 1
            )

    def calculate_metric_terms(self):
        """
        Compute the terms which are used for the metric
        """

        for diagnostic, observable in self.keys:
            key = (diagnostic, observable)

            experiment = self.experiment.get_observable(*key)

            simulation = self.simulation.get_observable(*key)
            if simulation.is_empty:
                continue

            comparison_data = compute_comparison_data(experiment, simulation)

            self.distance[diagnostic][observable] = calculate_normalised_ricci_distance(
                **comparison_data
            )

            self.sensitivity[diagnostic][observable] = calculate_ricci_sensitivity(
                **comparison_data
            )

    def compute_chi(
        self, agreement_threshold: float = 1.0, transition_sharpness: float = 0.5
    ):
        """
        Calculates the 'chi' validation metric and 'Q' quality factor
        """
        chi_numerator = 0.0
        chi_denominator = 0.0

        for diagnostic, observable in self.keys:

            distance = self.distance[diagnostic][observable]
            hierarchy = self.hierarchy[diagnostic][observable]
            sensitivity = self.sensitivity[diagnostic][observable]

            if np.any(np.isnan(distance)) or np.any(np.isnan(sensitivity)):
                # Penalise missing data. This is a harsh penalty, but acts as a strong disincentive to
                # drop data which would otherwise increase a comparisons chi. This way, including any data is guaranteed
                # to improve the chi value compared to leaving a field blank.
                R, S, H = 1.0, 1.0, 1.0
            else:
                R = level_of_agreement_function(
                    distance, agreement_threshold, transition_sharpness
                )
                S = sensitivity
                H = 1.0 / hierarchy

            chi_numerator += R * H * S
            chi_denominator += H * S

        chi = chi_numerator / chi_denominator
        Q = chi_denominator

        return chi, Q

    def as_dataframe(
        self, agreement_threshold: float = 1.0, transition_sharpness: float = 0.5
    ):
        """
        Returns a dataframe of the calculation
        """

        Row = namedtuple("Row", ("distance", "hierarchy", "sensitivity"))

        diagnostics, observables, rows = [], [], []

        for diagnostic, observable in self.keys:
            d = self.distance[diagnostic][observable]
            H = self.hierarchy[diagnostic][observable]
            S = self.sensitivity[diagnostic][observable]

            diagnostics.append(diagnostic)
            observables.append(observable)
            rows.append(Row(d, H, S))

        index = pd.MultiIndex.from_tuples(
            list(zip(diagnostics, observables)), names=["diagnostic", "observable"]
        )

        df = pd.DataFrame(rows, index=index)

        return df
