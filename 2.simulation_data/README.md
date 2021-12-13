# Simulation data

In this folder, you can find the data corresponding to simulations validated against the TCV-X21 reference scenario.

Additionally, checkpoint files from the GRILLIX 1mm simulations are provided in `GRILLIX_2021/checkpoints_for_1mm`. See `notebooks/simulation_postprocessing.ipynb` for an example of how to interface with these files. If you are having issues initialising a simulation, these could potentially offer a numerically-stable state (although please note that we do not make any statements about whether these checkpoints are physically sensible).

If you perform your own validation against TCV-X21, we'd be happy to include your results in the repository. The easiest way to do this is to generate your own folder with NetCDF files (see `tcvx21/record_c` for routines to write these files). Then, submit a merge request to the git repository. Feel free to include the post-processing routines as well.
