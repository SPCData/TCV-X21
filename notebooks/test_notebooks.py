import pytest
from pathlib import Path
import runpy
import os
import contextlib
from notebooks.notebooks_m import notebooks, notebook_dir
import matplotlib.pyplot as plt


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@pytest.mark.parametrize("filepath", notebooks.values(), ids=notebooks.keys())
def test_notebook(filepath):
    """
    Runs the python notebook associated with each .ipynb notebook.

    Make sure that you have executed 'nbsync' at the command line before running the tests!
    """
    print(notebook_dir)
    assert filepath.with_suffix(".py").exists()

    if filepath.stem == "failing_notebook":
        with pytest.raises(AssertionError):
            with working_directory(filepath.parent):
                runpy.run_path(str(filepath.with_suffix(".py")))
    else:
        with working_directory(filepath.parent):
            runpy.run_path(str(filepath.with_suffix(".py")))

    # Clean up all figures
    plt.close("all")
