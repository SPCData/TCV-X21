# TCV-X21 validation for divertor turbulence simulations

## Quick links

<!-- [![Paper](https://img.shields.io/badge/Paper-Nuclear%20Fusion-critical)](https://iopscience.iop.org/journal/0029-5515) -->
[![arXiv](http://img.shields.io/badge/arXiv-arXiv%3A2109.01618-B31B1B.svg)](https://arxiv.org/abs/2109.01618)
[![PDF](https://img.shields.io/badge/PDF-Oliveira%20%26%20Body%20et%20al.%202021-important)](2109.01618.pdf)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SPCData/TCV-X21/HEAD?labpath=tcv-x21.ipynb)
[![DOI](https://zenodo.org/badge/437798156.svg)](https://zenodo.org/badge/latestdoi/437798156)

[![Dataset licence](https://img.shields.io/badge/Dataset%20License-CC--BY--4.0-brightgreen)](LICENSE)
[![Software licence](https://img.shields.io/badge/Software%20License-MIT-brightgreen)](tcvx21/LICENSE)

[![Test Python package](https://github.com/SPCData/TCV-X21/actions/workflows/test_python_package.yml/badge.svg?branch=main)](https://github.com/SPCData/TCV-X21/actions/workflows/test_python_package.yml)
[![codecov](https://codecov.io/gh/SPCData/TCV-X21/branch/main/graph/badge.svg?token=mPj5fc8EX3)](https://codecov.io/gh/SPCData/TCV-X21)

## Intro

Welcome to TCV-X21. We're glad you've found us!

This repository is designed to let you perform the analysis presented in *Oliveira and Body et. al., Nuclear Fusion, 2021*, both using the data given in the paper, and with a turbulence simulation of your own. We hope that, by providing the analysis, the TCV-X21 case can be used as a standard validation and bench-marking case for turbulence simulations of the divertor in fusion experiments. The repository allows you to scrutinise and suggest improvements to the analysis (there's always room for improvement), to directly interact with and explore the data in greater depth than is possible in a paper, and — we hope — use this case to test a simulation of your own.

To use this repository, you'll need to either use the `mybinder.org` link below OR user rights on a computer with Python-3, conda and git-lfs pre-installed.

## Video tutorial

This quick tutorial shows you how to navigate the repository and use some of the functionality of the library.

## What can you find in this repository

* `1.experimental_data`: data from the TCV experimental campaign, in NetCDF, MATLAB and IMAS formats, as well as information about the reference scenario, and the reference magnetic geometry (in `.eqdsk`, `IMAS` and `PARALLAX-nc` formats)
* `2.simulation_data`: data from simulations of the TCV-X21 case, in NetCDF format, as well as raw data files and conversion routines
* `3.results`: high resolution PNGs and LaTeX-ready tables for a paper
* `tcvx21`: a Python library of software, which includes
  * `record_c`: a class to interface with NetCDF/HDF5 formatted data files
  * `observable_c`: a class to interact with and plot observables
  * `file_io`: tools to interact with MATLAB and JSON files
  * `quant_validation`: routines to perform the quantitative validation
  * `analysis`: statistics, curve-fitting, bootstrap algorithms, contour finding
  * `units_m.py`: setting up `pint`-based unit-aware analysis (it's difficult to overstate how cool this library is)
  * `grillix_post`: a set of routines used for post-processing GRILLIX simulation data, which might help if you're trying to post-process your own simulation. You can see a worked example in `simulation_postprocessing.ipynb`
* `notebooks`: Jupyter notebooks, which allow us to provide code with outputs and comments together
  * `simulation_setup.ipynb`: what you might need to set up a simulation to test
  * `simulation_postprocessing.ipynb`: how to post-process the data
  * `data_exploration.ipynb`: some examples to get you started exploring the data
  * `bulk_process.ipynb`: runs over every observable to make the `results` — which you'll need to do if you're writing a paper from the results
* `tests`: tests to make sure that we haven't broken anything in the analysis routines
* `README.md`: this file, which helps you to get the software up and running, and to explain where you can find everything you need. It also provides the details of the licencing (below). There's more specific `README.md` files in several of the subfolders.

and lots more files. If you're not a developer, you can safely ignore these.

## What can't you find in this repository

Due to licencing issues, the source code of the simulations is *not provided*. Sorry!

Also, the raw simulations are not provided here due to space limitations (some runs have more than a terabyte of data), but they are all backed up on archive servers. If you'd like to access the raw data, get in contact.

## License and attribution notice

The TCV-X21 datasets are licenced under a Creative Commons Attribution 4.0 license, given in `LICENCE`. The source code of the analysis routines and Python library is licenced under a MIT license, given in `tcvx21/LICENCE`.

For the datasets, we ask that you provide attribution if using this data via the citation in the `CITATION.cff` file. We additionally require that you mark any changes to the dataset, and state specifically that the authors do not endorse your work unless such endorsement has been expressly given.

For the software, you can use, modify and share without attribution or marking changes.

## Running the Jupyter notebooks (installation as non-root user)

To run the Jupyter notebooks, you have two options. The first is to use the `mybinder.org` interface, which let you interact with the notebooks via a web interface. You can launch the binder for this repository by clicking the binder badge in the repository header. Note that not all of the
repository content is copied to the Docker image (this is specified in `.dockerignore`). The large checkpoint files are not
included in the image, although they can be found in the repository at `2.simulation_data/GRILLIX/checkpoints_for_1mm`.
Additionally, the default docker image will not work with git.

Alternatively, if you'd like to run the notebooks locally or to extend the repository, you'll need to *install* additional Python packages. First of all, you need Python-3 and conda installed (latest versions recommended). Then, to install the necessary packages, we make a sandbox environment. This has a few advantages to installing packages globally — sudo rights are not required, you can install package versions without risking breaking other Python scripts, and if everything goes terribly wrong you can easily delete everything and restart. We've included a simple shell script to perform the necessary steps, which you can execute with

```bash
./install_env.sh
```

This will install the library in a subfolder of the TCV-X21 repository called `tcvx21_env`. It will also add a kernel to your global Jupyter installation. To remove the repository, you can delete the folder `tcvx21_env` and run `jupyter kernelspec uninstall tcvx21`.

### To run tests and open Jupyter

Once you've installed via either option, you can activate the python environment with `conda activate ./tcvx21_env`. To deactivate, run `conda deactivate`.

Then, it is recommended to run the test suite with `pytest` which
ensures that everything is installed and working correctly. If something fails, let us know in the issues. Note that this executes all of the analysis notebooks, so it might take a while to run.

Finally, run `jupyter lab` to open a Jupyter server in the TCV-X21 repository. Then, you can open any of the notebooks (`.ipynb` extension) by clicking in the side-bar.

### A note on pinned dependencies

To ensure that the results are reproducible, the `environment.yml` file has pinned dependencies. However, if you want to use
this software as a library, pinned dependencies are unnecessarily restrictive. You can remove the versions after the `=` sign in the `environment.yml`, but be warned that things might break.
