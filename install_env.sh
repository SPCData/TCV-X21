#!/usr/bin/env bash

# Simple command line utility to install the tcvx21 environment and install jupyter requirements
# If an command-line-argument "clean" is given, then the script will clean and reinstall the environment.
# If it is anything else (or empty), the script will use the existing environment and
# only set up console scripts and the Jupyter environment

# Must be run from within the tcvx21 top-level directory (i.e. where this script is saved)
script_name=$0
script_full_path=$(dirname "$0")

echo "script_name: $script_name"
echo "full path: $script_full_path"

if [[ ${script_full_path} != "." ]];then
    echo "Setup script must be run locally (i.e. path = .). Exiting."
    exit 1
fi

echo "Loading module anaconda"
set +e
module load anaconda
set -e

start=`date +%s`
CLEAN=$1

echo "environment name: ./tcvx21_env"
if [[ ${CLEAN} == "clean" ]]; then
    echo "Cleaning ./tcvx21_env (ctrl-D in 5s to cancel)"
    # sleep 5s
    echo "Continuing"
else
    echo "Will not clean"
fi

# Check if the environment already exits
if [ -d ./tcvx21_env ]; then
    if [[ ${CLEAN} == "clean" ]]; then
        echo "Cleaning and rebuilding environment for ./tcvx21_env"
        conda remove --prefix ./tcvx21_env --all -y
        conda env create --prefix ./tcvx21_env --file=environment.yml
    else
        echo "Found existing environment for ./tcvx21_env. Use `./install_env.sh` clean to clean install"
    fi
else
    echo "No installation found for ./tcvx21_env. Building."
    conda env create --prefix ./tcvx21_env --file=environment.yml
fi

# Activate the virtual environment
source activate ./tcvx21_env
echo "Unloading module anaconda"
set +e
module unload anaconda
set -e

WHICH_PYTHON=$(which python)
echo "Using python=$WHICH_PYTHON"
if grep -q ./tcvx21_env <<< "$WHICH_PYTHON"; then
    echo "Activated ./tcvx21_env"
else
    echo "Failed to activate environment ./tcvx21_env"
    exit 1
fi

# Install the tcvx21 module
echo "Installing tcvx21 module"
pip install --editable .

# Install the jupyter kernel
echo "Installing the tcvx21 Jupyter kernel"
echo "If this fails, try append the following to your ~/.bashrc and running source ~/.bashrc"
echo "export JUPYTERLAB_DIR=\"$HOME/.local/share/jupyter/lab\""
python -m ipykernel install --user --name tcv-x21

# Install the pre-commit hook for storing notebooks as text files
nbsync

end=`date +%s`
runtime=$((end-start))
echo "Finished in $runtime s"
