# This file helps to simplify import paths -- so you can type `import tcvx21.read_from_json` instead of
# `from tcvx21.json_io_m import read_from_json`
# Only key functionality is imported here -- use the full path to access non-key functionality

from pathlib import Path

repository_dir = Path(__file__).parents[1]
library_dir = Path(__file__).parent
sample_data = repository_dir / "tests" / "sample_data"

experimental_reference_dir = repository_dir / "1.experimental_data"
divertor_coords_json = (
    experimental_reference_dir / "reference_scenario/divertor_polygon.json"
)
rdpa_coords_json = (
    experimental_reference_dir / "reference_scenario/RDPA_reference_coords.json"
)
thomson_coords_json = (
    experimental_reference_dir / "reference_scenario/thomson_position.json"
)

simulation_dir = repository_dir / "2.simulation_data"
grillix_dir = simulation_dir / "GRILLIX_2021"
grillix_forward_dir = grillix_dir / "checkpoints_for_1mm/forward_field"
grillix_reversed_dir = grillix_dir / "checkpoints_for_1mm/reversed_field"
gbs_dir = simulation_dir / "GBS_2021"
tokam3x_dir = simulation_dir / "TOKAM3X_2021"

notebooks_dir = repository_dir / "notebooks"
test_dir = repository_dir / "tests"
results_dir = repository_dir / "3.results"

test_figures_dir = results_dir / "test_fig"

for directory in [
    repository_dir,
    library_dir,
    experimental_reference_dir,
    simulation_dir,
    notebooks_dir,
    test_dir,
]:
    assert directory.exists() and directory.is_dir(), f"{directory} is not a directory"

style_sheet = library_dir / "tcvx21.mplstyle"
style_sheet_inline = library_dir / "tcvx21_inline.mplstyle"
assert style_sheet.exists()

import sys

test_session = "pytest" in sys.modules

from .units_m import Quantity, Dimensionless, convert_xarray_to_quantity, unit_registry
from .file_io import read_from_json, write_to_json, read_struct_from_file
from .record_c import Record, EmptyRecord, template_file
from .dict_methods_m import (
    summarise_tree_dict,
    recursive_rename_keys,
    recursive_assert_dictionaries_equal,
)


# Add subdirectories to namespace (can be called as tcvx21.grillix_post)
from . import grillix_post
from . import analysis
from . import file_io
from . import observable_c
