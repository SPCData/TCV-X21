# TCV-X21 FAIR repository

This repository aims to ensure that the TCV-X21 validation data is FAIR — Findable, Accessible, Interoperable and Reproducible. The repository accompanies the article Oliveira and Body et. al., 2021, *“Validation of edge turbulence codes against the TCV-X21 diverted L-mode reference case”*, (submitted to Nuclear Fusion as an Open Access article). You can find a copy of the paper in the repository

The repository is intended to

1. Enable non-affiliated groups to reproduce the results of the validation with their own turbulence code. For this purpose, the simulation inputs are included (magnetic equilibrium, recommended source positions), as are example post-processing scripts and sample data from GRILLIX.
2. Allow readers of the paper to verify and scrutinise the post-processing procedure, to ensure that the validation result is fair and reproducible. For this purpose, the post-processed profiles from each code and from TCV are included, as are Python routines which plot the results and calculate the validation metric.
3. Permit future validations (including by non-affiliated groups) to include their results for comparitive validation and benchmarking. Additionally, allow the experimental dataset to be extended with new results. For this purpose, links to both a static repository (via Zenodo) and a dynamic repository (via GitHub) should be included in any work detailing TCV-X21 results.

The repository includes

1. A copy of Oliveira and Body *et al.,* 2021, *Validation of edge turbulence codes against the TCV-X21 diverted L-mode reference case*
2. High resolution PNG figures of all comparisons, as well as various summary and analysis figures
3. The ‘reference’ magnetic equilibrium, in eqdsk and IMAS formats as well as custom formats
4. The analytic form and magnitude of the sources used in the paper
5. The profiles of the following TCV results (as functions of $R^u - R^u_{sep}$ for 1D profiles, and $R^u - R^u_{sep}, Z - Z_X$ for 2D profiles), provided as NetCDF4/HDF5 files
   1. Mean profiles for the plasma density, electron temperature, plasma potential, ion saturation current, floating potential and parallel current density measured by the wall Langmuir probes
   2. The standard deviation of the ion saturation current, floating potential and parallel current density measured by the wall Langmuir probes
   3. The skewness and kurtosis of the ion saturation current, measured by the wall Langmuir probes
   4. The parallel heat flux at the low field side target, measured by the infrared camera
   5. Mean profiles for the plasma density, electron temperature, plasma potential, ion saturation current, floating potential and parallel Mach number measured by the reciprocating divertor probe array
   6. The standard deviation of the ion saturation current measured by the reciprocating divertor probe array
   7. Mean profiles of the plasma density and electron temperature measured by Thomson scattering
   8. Mean profiles for the plasma density, electron temperature, plasma potential, ion saturation current, floating potential and parallel Mach number measured by the reciprocating midplane probe
   9. The standard deviation of the ion saturation current, floating potential and parallel current density measured by the reciprocating midplane probe
   10. The skewness and kurtosis of the ion saturation current, measured by the reciprocating midplane probe
6. An IMAS-compatible version of the TCV results
7. The $R, Z$ positions of the reciprocating divertor probe array and the Thomson scattering diagnostic
8. The profiles obtained by GBS, GRILLIX and TOKAM3X, from simulations of the reference equilibrium (performed in two field directions, except for TOKAM3X which provides forward-field data only), provided as a NetCDF4/HDF5 file per code and per field direction
9. 2D profiles of the plasma density from GBS, GRILLIX and TOKAM3X for a single plane, a single time point and a single toroidal field direction
10. From GRILLIX only, 2D profiles of the plasma density, electron and ion temperatures, generalised vorticity, parallel ion velocity, parallel current density, electrostatic potential and the parallel component of the electromagnetic potential for all planes, for a single time point, for two field directions— both to demonstrate the post-processing routines on, and to optionally be used as initial conditions for future simulations
11. A “data template” file in both `.json` and `.mat` format, indicating all measurements required for performing the validation against the TCV-X21 reference case
12. A Python library allowing
    1. A user to write a NetCDF4/HDF5 file from a dictionary of results
    2. Interfacing with results stored in a NetCDF4/HDF5 file, including plotting routines for 1D and 2D data and automatic handling of units
    3. Routines to calculate the validation metric presented in Ricci et. al., 2015 from a set of NetCDF4/HDF5 files
    4. Routines for reproducing figures of all results in the paper
    5. Routines for extracting the validation results from raw GRILLIX data, including volume integrators for the power and particle sources, interpolation routines for extracting data at arbitrary $R, Z$ positions, routines for computing experimental data ($J_{sat}, V_{fl}, q_\parallel, M_\parallel$) and methods for computing normalisation and physical parameters in SI units.
13. Jupyter notebooks providing worked examples for
    1. Interfacing with the equilibrium file and setting the sources
    2. Performing the post-processing for the GRILLIX example
    3. Performing the validation analysis as it is presented in the paper
