"""
Defines a command-line function which performs a similar function to
jupytext --sync <notebooks>

However, this version already knows the notebooks which should be updated
(based on the notebooks_m file)
"""
from jupytext.cli import jupytext
from .notebooks_m import notebooks
import click
from pathlib import Path

def sync_files(filepath: Path):
    if not filepath.exists():
        filepath = filepath.with_suffix('.py')
        assert filepath.exists()

    jupytext(["--sync", f"{filepath}", "--pipe", "black"])

@click.command()
@click.option('--key', '-k',
              type=click.Choice(list(notebooks.keys())),
              default=None,
              help="Act on a single file only (according to the key)")
def sync_notebooks(key: str=None):
    """
    Synchronises the py:percent and ipynb notebook representations
    If only one of the two files is present, this function builds the other file
    If both files are present, both files are set to whichever is newer of the script or 
    ipynb representation

    WARNING: this function can make irreversible changes and delete your work. Always edit
    ONLY one of the .py or .ipynb, and be very careful if you have uncommitted work on
    either file.

    There is no diff functionality, so if you're only working on a single file you can specify a
    file key (corresponding to the entry in notebooks_m) via the -k flag

    This will update files without asking for confirmation

    Finally, there's no way to select the older file. If you want the older file, delete the newer one
    """

    if key is not None:
        
        sync_files(notebooks[key])
        
    else:
        for filepath in notebooks.values():
            sync_files(filepath)

if __name__ == "__main__":
    sync_notebooks()
