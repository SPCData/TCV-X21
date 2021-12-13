# Reference scenario details

In this folder you can find several files which provide additional details about the TCV-X21 reference scenario.

* `65402_t1.eqdsk` is the magnetic equilibrium of TCV discharge 65402 at time t = 1.0 seconds. It is the equilibrium used by the simulations for modelling. This is the "reference equilibrium".
* `reference_equilibrium.nc` is a PARALLAX-formatted equilibrium NetCDF, which may be easier to use if you already have a NetCDF reader.
* `dataset_TCV_CORE.mat` gives values measured in the core of TCV during one of the TCV-X21 discharges. It is currently not officially part of the TCV-X21 dataset, but may be helpful for setting up an initial condition in a simulation.
* `divertor_polygon.json` gives the R, Z points of a 2D polygon, which approximates the TCV wall and divertor (without baffles).
* `physical_parameters.nml` gives the recommended normalisation parameters for a simulation.
* `RDPA_coordinates.ipynb` converts the $`R^u-R^u_{sep}, Z - Z_X`$ RDPA coordinates into $`R, Z`$ coordinates in the reference equilibrium. It produces the `RDPA_reference_coords.json` file as output.
* `thomson_position.json` gives the sample positions of the Thomson Scattering system.
