"""
This file defines all of the notebooks tested with pytest and synced with nbsync
"""
from pathlib import Path
from tcvx21 import experimental_reference_dir, gbs_dir, tokam3x_dir

notebook_dir = Path(__file__).parent.absolute()

notebooks = dict(
    failing_notebook=(notebook_dir / "failing_notebook.ipynb"),
    landing_page=(notebook_dir.parent / "tcv-x21.ipynb"),
    simulation_setup=(notebook_dir / "simulation_setup.ipynb"),
    data_exploration=(notebook_dir / "data_exploration.ipynb"),
    bulk_process=(notebook_dir / "bulk_process.ipynb"),
    TCV_processing=(experimental_reference_dir / "TCV_processing.ipynb"),
    RDPA_coordinates=(
        experimental_reference_dir / "reference_scenario" / "RDPA_coordinates.ipynb"
    ),
    GBS_processing=(gbs_dir / "GBS_processing.ipynb"),
    simulation_postprocessing=(notebook_dir / "simulation_postprocessing.ipynb"),
    TOKAM3X_processing=(tokam3x_dir / "TOKAM3X_processing.ipynb"),
)
