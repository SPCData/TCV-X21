# Notebooks

Jupyter notebooks showing some of the validation analysis. If you'd like to interact with these notebooks, we recommend using the binder service (see the badge in the top-level `README.md`). This allows you to interact with the notebooks via a web service.

Alternatively, you can follow the installation instructions in the README and launch a jupyter lab session via `jupyter lab` in the terminal.

## Testing and making changes

To make it easier to check changes and perform unit testing on the jupyter notebooks, we use the `jupytext` package to convert the notebook files to Python representations. To convert the notebooks listed in `notebooks_m` to/from `.py` files, use the `nbsync` tool on the command-line.
