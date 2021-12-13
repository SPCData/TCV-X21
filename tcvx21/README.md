# Python analysis routines for TCV-X21

This folder contains a reasonably extensive analysis library. These are the routines used for the analysis in the TCV-X21 paper, and they are included here in case you find them useful.

There is a lot of functionality, so unfortunately if you're not familiar with Python these might be a bit difficult to pick up. However, feel free to use the discussion board or contact the authors for a tour.

The routines in this folder (and the surrounding library and testing infrastructure) are licenced under an MIT licence.


## Navigating for developers

* `record_c`: a class to wrap the standard NetCDF files
* `observable_c`: a class to interface with single observables
* `units_m.py`: an interface to Pint (unit-conversion)
* `plotting`: all of the routines need to make the figures in `results`. Note that, at the lowest level, the plot command eventually calls a `.plot` method on the observable class -- since 1D and 2D observables are handled differently.
* `grillix_post`: an extensive post-processing library for GRILLIX data, which may help if you need to write your own post-processing library.
