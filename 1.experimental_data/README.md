# Experimental data

This folder contains the experimental data for the TCV-X21 validation.
The analysis (routines in the `tcvx21` folder) uses the `.nc` files. These are NetCDF files which match the format defined by `tcvx21.record_c.observables.json`. The experimental data is also available as a MATLAB file `dataset_TCVX21.mat`.

The structure of the NetCDF files is

1. A file per field direction
2. Within each file, a group per diagnostic
3. Within each diagnostic group, a "name" and an "observables" group
4. Within each observables group, the following structure:
    1. "name": a long-form name
    2. "values": the values recorded for the observable, with units "values:units"
    3. "errors": the uncertainty of the values, with units "errors:units"
    4. "Rsep_omp": the measurement position $`R^u - R^u_{sep}`$, with units "Rsep_omp:units"
    5. "Zx": (RDPA only) the measurement position $`Z - Z_X`$, with units "Zx:units"
    6. "experimental_hierarchy": the primacy hierarchy for the experimental observable (used in the Ricci validation methodology)
    7. "simulation_hierarchy": the primacy hierarchy for the simulated observable. A value of -1 is used as a placeholder in the experimental data

The IMAS-formatted data files are available in the IMAS_data subfolder.
Additional details about the TCV-X21 reference scenario are in the `reference_scenario` folder.
