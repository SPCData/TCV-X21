"""
Test the validation analysis
"""
import numpy as np
import pytest
from pathlib import Path
import tcvx21
from tcvx21 import quant_validation as qv


@pytest.fixture(scope="module")
def diagnostic():
    return "LFS-LP"


@pytest.fixture(scope="module")
def observable():
    return "density"


@pytest.fixture(scope="module")
def experimental_reference(tcv_forward_data):
    return tcvx21.Record(tcv_forward_data)


@pytest.fixture(scope="module")
def simulation_sample():
    return tcvx21.Record(tcvx21.test_dir / 'comparison_data' / "GRILLIX_example.nc")


@pytest.fixture(scope="module")
def expt_observable(experimental_reference, diagnostic, observable):
    return experimental_reference.get_observable(diagnostic, observable)


@pytest.fixture(scope="module")
def sim_observable(simulation_sample, diagnostic, observable):
    return simulation_sample.get_observable(diagnostic, observable)


@pytest.fixture(scope="module")
def comparison_data(expt_observable, sim_observable):
    comp_data = qv.compute_comparison_data(
        experiment=expt_observable, simulation=sim_observable
    )

    return comp_data


def test_normalised_distance(comparison_data, expt_observable, sim_observable):

    d1 = qv.calculate_normalised_ricci_distance(**comparison_data)
    d2 = qv.calculate_normalised_ricci_distance_from_observables(
        expt_observable, sim_observable
    )

    assert np.isclose(d1, d2)
    assert np.isclose(d1, 2.0981, rtol=1e-2)

    assert np.isclose(qv.level_of_agreement_function(d1), 0.923155, rtol=1e-2)


def test_normalised_distance_2D(experimental_reference, simulation_sample):
    d = qv.calculate_normalised_ricci_distance_from_observables(
        experiment=experimental_reference.get_observable("RDPA", "density"),
        simulation=simulation_sample.get_observable("RDPA", "density"),
    )

    assert np.isclose(d, 3.2298, rtol=1e-2)


def test_sensitivity(comparison_data):

    s1 = qv.calculate_ricci_sensitivity(**comparison_data)

    assert np.isclose(s1, 0.8444, rtol=1e-2)


def test_validation_class(
    experimental_reference, simulation_sample, diagnostic, observable, comparison_data
):

    validation = qv.RicciValidation(
        experimental_dataset=experimental_reference,
        simulation_dataset=simulation_sample,
    )

    validation.calculate_metric_terms()

    assert np.isclose(
        validation.distance[diagnostic][observable],
        qv.calculate_normalised_ricci_distance(**comparison_data),
    )
    assert np.isclose(
        validation.sensitivity[diagnostic][observable],
        qv.calculate_ricci_sensitivity(**comparison_data),
    )

    chi, Q = validation.compute_chi(agreement_threshold=1.0)
    assert np.isclose(chi, 0.78158, rtol=1e-2)
    assert np.isclose(Q, 16.6824, rtol=1e-2)

    chi2, Q2 = validation.compute_chi(agreement_threshold=5.0)
    assert chi2 < chi
    assert np.isclose(Q, Q2, rtol=1e-2)

    df = validation.as_dataframe()
    assert np.isclose(
        df["distance"][diagnostic, observable],
        qv.calculate_normalised_ricci_distance(**comparison_data),
        rtol=1e-2,
    )


def test_table_writer(experimental_reference, simulation_sample, tmpdir):

    validation = qv.RicciValidation(
        experimental_dataset=experimental_reference,
        simulation_dataset=simulation_sample,
    )

    validation.calculate_metric_terms()

    qv.write_cases_to_latex(
        cases={"example+": validation},
        output_file=Path(tmpdir / "validation_table.tex"),
    )
