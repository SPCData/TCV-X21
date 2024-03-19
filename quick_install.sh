# Install micromamba into a local directory
BIN_FOLDER="micromamba"
INIT_YES="no"
CONDA_FORGE_YES="no"
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

micromamba/micromamba create -p ./tcvx21_env -f environment.yml
micromamba/micromamba run -p ./tcvx21_env pip install -e .

export PATH="${PWD}/tcvx21_env/bin:$PATH"
./tcvx21_env/bin/nbsync