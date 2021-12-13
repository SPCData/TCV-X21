"""
Installs the Python package of processing routines
"""
from setuptools import setup, find_packages

setup(
    name="tcvx21",
    author="Thomas Body",
    version="1.0",
    description="Python tools for interacting with the TCV-X21 validation dataset",
    url="https://github.com/SPCData/TCV-X21",
    packages=find_packages(include=["tcvx21", "notebooks"]),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        nbsync=notebooks.sync_notebooks:sync_notebooks
    """,
)
