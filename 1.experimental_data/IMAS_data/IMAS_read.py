import imas
from imas import imasdef

import numpy as np

# Testing reading back the TCV-X21 dataset from IMAS format
# Author : F. Imbeaux, 2021

pulse = 10
run = 0
imas_entry = imas.DBEntry(imasdef.HDF5_BACKEND, "tcv", pulse, run)
imas_entry.open()

lfs_lp = imas_entry.get("langmuir_probes", 0)
print(lfs_lp.ids_properties)
print(len(lfs_lp.embedded))
print(lfs_lp.embedded[0].n_e.data)
