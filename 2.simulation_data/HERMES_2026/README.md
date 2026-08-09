# Hermes-3 TCV-X21 datasets

B.D. Dudson, M. Kryjak, H. Muhammed and J.T. Omotani
Validation of Hermes-3 turbulence simulations against the TCV-X21 diverted L-mode reference case
Nucl. Fusion 66 036015 (2026)

https://doi.org/10.1088/1741-4326/ae3627

TCV-X21 standard datasets in NetCDF format:
- Forward field: hermes3_forward_field.nc
- Reversed field: hermes3_reversed_field.nc

These datasets were generated using these steps:

1. Run Hermes-3 simulations using input files in the `forward_inputs`
   and `reversed_inputs` subdirectories. Simulations were performed
   with Hermes-3 version 05d4d98 linked to BOUT++ version a776c8c.
   Source code is available at https://github.com/boutproject/hermes-3
   and manual at https://hermes3.readthedocs.io/en/latest/.

2. Gather simulation data into pickle files using `gather_data.py`
   Simulations are usually performed in multiple "runs", due to HPC
   facility limits on runtime and the size of the output files.  After
   the simulation has reached quasi-steady state, each simulation
   dataset should be gathered into a pickle file. The `gather_data.py`
   script can concatenate these files into a single pickle file.

3. Convert the pickle file into a standard TCV-X21 NetCDF file using
   `convert_to_tcvx21.py`. This adds metadata so that the data is
   compatible with the TCV-X21 analysis tools in
   https://github.com/SPCdata/TCV-X21.

LLNL dataset 2026-056
