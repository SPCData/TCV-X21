"""
Defines pytest fixtures
"""
import pytest
from pathlib import Path
import tcvx21
from tcvx21 import test_dir


@pytest.fixture(scope="session")
def tcv_forward_data() -> Path:
    return test_dir / "comparison_data" / "TCV_forward_field.nc"


@pytest.fixture(scope="session")
def tcv_reversed_data() -> Path:
    return test_dir / "comparison_data" / "TCV_reversed_field.nc"


@pytest.fixture(scope="session")
def grx_forward_data() -> Path:
    return test_dir / "comparison_data" / "GRILLIX_forward_field.nc"


@pytest.fixture(scope="session")
def grx_reversed_data() -> Path:
    return test_dir / "comparison_data" / "GRILLIX_reversed_field.nc"


@pytest.fixture(scope="session")
def comparison_data(
    tcv_forward_data, tcv_reversed_data, grx_forward_data, grx_reversed_data
):

    return dict(
        experimental_data=dict(
            forward_field=tcvx21.Record(tcv_forward_data, color="C0", linestyle="-"),
            reversed_field=tcvx21.Record(tcv_reversed_data, color="C0", linestyle="-"),
        ),
        simulation_data=dict(
            GRILLIX=dict(
                forward_field=tcvx21.Record(
                    grx_forward_data, color="C1", linestyle="--"
                ),
                reversed_field=tcvx21.Record(
                    grx_reversed_data, color="C1", linestyle="--"
                ),
            ),
        ),
    )
