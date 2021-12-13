# Dockerfile for mybinder.org.
# 
# Use the pre-built Jupyter-docker-stack.
# This is a docker image with a significant scientific software stack preinstalled
# See https://jupyter-docker-stacks.readthedocs.io/en/latest/
FROM jupyter/scipy-notebook:7aa954ab78d1

# Update the 'base' conda environment of the docker image with the additional requirements for the tcvx21
# library. We use the exact versions of python libraries included in the docker image, so that the update
# only extends the environment with additional packages rather than reinstalling.
COPY environment.yml .
RUN conda env update --file environment.yml --name base && \ 
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy the rest of the source from the repository to the docker image.
# Note that we use .dockerignore to ignore '.git' (since history can add to the image size) and './env' (since, if you've
# locally installed the tcvx21 environment, you don't want to copy this to the image -- it's also quite big)
COPY --chown=$NB_USER:$NB_GID . ./

# Install the 'tcvx21' library
RUN pip install --no-cache --editable .

# Build the notebook .ipynb representations
RUN nbsync

# The rest of the settings should be left unchanged (since the Jupyter-Docker-Stacks are designed to work with mybinder.org)
