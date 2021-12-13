import imas
from imas import imasdef

import numpy as np
from netCDF4 import Dataset

# Mapping the TCV-X21 dataset to IMAS format
# Author : F. Imbeaux, 2021
# General description of the dataset --> dataset_description IDS, occurrence 0
# Diagnostic LFS-LP --> langmuir_probes IDS, occurrence 0
# Diagnostic HFS-LP --> langmuir_probes IDS, occurrence 1
# Diagnostic LFS-IR --> camera_ir IDS, occurrence 0
# Diagnostic FHRP --> langmuir_probes IDS, occurrence 2
# Diagnostic RPDA --> langmuir_probes IDS, occurrence 3
# Diagnostic TS --> thomson_scattering IDS, occurrence 0


# Open the original netCDF files
tcv_data = Dataset("./TCV_forward_field.nc")
# ne = tcv_data['LFS-LP/observables/density']
# print(ne['value'][:])
# print(ne['Rsep_omp'][:])  # Caution, Rsep units are cm !

# plt.errorbar(jsat_lfs['Rsep_omp'][:], jsat_lfs['value'][:], jsat_lfs['error'][:])

# Create the output file
pulse = 10
run = 0
imas_entry = imas.DBEntry(imasdef.HDF5_BACKEND, "tcv", pulse, run)
imas_entry.create()

##################### Dataset description
dd = imas.dataset_description()
dd.ids_properties.homogeneous_time = 1
dd.ids_properties.comment = (
    "TCV-X21 dataset, for forward field. Data has been processed over multiple pulses and time slices, and mapped onto the distance to separatrix at outboard midplane Rsep_omp (distance_separatrix_midplane in IMAS). Due to this process, langmuir probes array indices in IMAS don"
    "t correspond to real probes but rather to a given Rsep_omp position of measurement collected over this multi-pulse dataset. Only some physical quantities for Langmuir probes are processed at a given position, e.g. electron density is not recorded at the same Rsep_omp positions as the saturation current, so they are recorded in different indices of the "
    "embedded"
    " or of the "
    "reciprocating"
    " array of structure in IMAS"
)
dd.ids_properties.source = "TCV_forward_field.nc"
dd.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
dd.dd_version = "3.33.0"
dd.time = np.array([0.0])  # Time has no meaning for this IDS

# IDS variable is filled, we write it now to the data entry
imas_entry.put(dd, 0)


##################### LFS-LP data
lfs_lp = imas.langmuir_probes()
lfs_lp.ids_properties.homogeneous_time = 1
lfs_lp.ids_properties.comment = tcv_data["LFS-LP"].diagnostic_name
lfs_lp.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
lfs_lp.time = np.array(
    [0.0]
)  # Time has no meaning for this dataset which is processed over several pulses and time slices
# midplane definition
lfs_lp.midplane.name = "dr_dz_zero_sep"
lfs_lp.midplane.index = 2
lfs_lp.midplane.description = "Midplane defined by the height of the outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen. equilibrium/time_slice/boundary_separatrix/dr_dz_zero_point/z"
# Different physical data have been gathered during different group of TCV pulses, therefore they are also measured at different locations. Typically : density is not measured at the same Rsep_omp positions as jsat. For each set of measurement positions, we create a set of indices in the "embedded" AoS, since this one assumes that a given probe is located at given position
# Get number of channels for density/electron_temp/potential group
channels_n = len(tcv_data["LFS-LP/observables/density/value"])
# Get number of channels for current/current_std group
channels_c = len(tcv_data["LFS-LP/observables/current/value"])
# Get number of channels for jsat/jsat_std/jsat_skew/jsat_kurtosis group
channels_j = len(tcv_data["LFS-LP/observables/jsat/value"])
# Get number of channels for vfloat/vfloat_std group
channels_v = len(tcv_data["LFS-LP/observables/vfloat/value"])

lfs_lp.embedded.resize(channels_n + channels_c + channels_j + channels_v)

for channel in range(channels_n):
    print(channel)
    # Positions
    lfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [tcv_data["LFS-LP/observables/density/Rsep_omp"][channel] / 100.0]
    )
    # Electron density
    lfs_lp.embedded[channel].n_e.data = np.array(
        [tcv_data["LFS-LP/observables/density/value"][channel]]
    )
    lfs_lp.embedded[channel].n_e.data_error_upper = np.array(
        [tcv_data["LFS-LP/observables/density/error"][channel]]
    )
    # Electron temperature
    lfs_lp.embedded[channel].t_e.data = np.array(
        [tcv_data["LFS-LP/observables/electron_temp/value"][channel]]
    )
    lfs_lp.embedded[channel].t_e.data_error_upper = np.array(
        [tcv_data["LFS-LP/observables/electron_temp/error"][channel]]
    )
    # Plasma potential
    lfs_lp.embedded[channel].v_plasma.data = np.array(
        [tcv_data["LFS-LP/observables/potential/value"][channel]]
    )
    lfs_lp.embedded[channel].v_plasma.data_error_upper = np.array(
        [tcv_data["LFS-LP/observables/potential/error"][channel]]
    )

for channel in range(channels_n, channels_n + channels_c):
    print(channel)
    # Positions
    lfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [tcv_data["LFS-LP/observables/current/Rsep_omp"][channel - channels_n] / 100.0]
    )
    # Parallel current density
    lfs_lp.embedded[channel].j_i_parallel.data = np.array(
        [tcv_data["LFS-LP/observables/current/value"][channel - channels_n]]
    )
    lfs_lp.embedded[channel].j_i_parallel.data_error_upper = np.array(
        [tcv_data["LFS-LP/observables/current/error"][channel - channels_n]]
    )
    # Parallel current density standard deviation
    lfs_lp.embedded[channel].j_i_parallel_sigma.data = np.array(
        [tcv_data["LFS-LP/observables/current_std/value"][channel - channels_n]]
    )
    lfs_lp.embedded[channel].j_i_parallel_sigma.data_error_upper = np.array(
        [tcv_data["LFS-LP/observables/current_std/error"][channel - channels_n]]
    )

for channel in range(channels_n + channels_c, channels_n + channels_c + channels_j):
    print(channel)
    # Positions
    lfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [
            tcv_data["LFS-LP/observables/jsat/Rsep_omp"][
                channel - channels_n - channels_c
            ]
            / 100.0
        ]
    )
    # Ion saturation current density
    lfs_lp.embedded[channel].j_i_saturation.data = np.array(
        [tcv_data["LFS-LP/observables/jsat/value"][channel - channels_n - channels_c]]
    )
    lfs_lp.embedded[channel].j_i_saturation.data_error_upper = np.array(
        [tcv_data["LFS-LP/observables/jsat/error"][channel - channels_n - channels_c]]
    )
    # Ion saturation current density standard deviation
    lfs_lp.embedded[channel].j_i_saturation_sigma.data = np.array(
        [
            tcv_data["LFS-LP/observables/jsat_std/value"][
                channel - channels_n - channels_c
            ]
        ]
    )
    lfs_lp.embedded[channel].j_i_saturation_sigma.data_error_upper = np.array(
        [
            tcv_data["LFS-LP/observables/jsat_std/error"][
                channel - channels_n - channels_c
            ]
        ]
    )
    # Ion saturation current density skew
    lfs_lp.embedded[channel].j_i_saturation_skew.data = np.array(
        [
            tcv_data["LFS-LP/observables/jsat_skew/value"][
                channel - channels_n - channels_c
            ]
        ]
    )
    lfs_lp.embedded[channel].j_i_saturation_skew.data_error_upper = np.array(
        [
            tcv_data["LFS-LP/observables/jsat_skew/error"][
                channel - channels_n - channels_c
            ]
        ]
    )
    # Ion saturation current density kurtosis
    lfs_lp.embedded[channel].j_i_saturation_kurtosis.data = np.array(
        [
            tcv_data["LFS-LP/observables/jsat_kurtosis/value"][
                channel - channels_n - channels_c
            ]
        ]
    )
    lfs_lp.embedded[channel].j_i_saturation_kurtosis.data_error_upper = np.array(
        [
            tcv_data["LFS-LP/observables/jsat_kurtosis/error"][
                channel - channels_n - channels_c
            ]
        ]
    )

for channel in range(
    channels_n + channels_c + channels_j,
    channels_n + channels_c + channels_j + channels_v,
):
    print(channel)
    # Positions
    lfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [
            tcv_data["LFS-LP/observables/vfloat/Rsep_omp"][
                channel - channels_n - channels_c - channels_j
            ]
            / 100.0
        ]
    )
    # Floating potential
    lfs_lp.embedded[channel].v_floating.data = np.array(
        [
            tcv_data["LFS-LP/observables/vfloat/value"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )
    lfs_lp.embedded[channel].v_floating.data_error_upper = np.array(
        [
            tcv_data["LFS-LP/observables/vfloat/error"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )
    # Floating potential standard deviation
    lfs_lp.embedded[channel].v_floating_sigma.data = np.array(
        [
            tcv_data["LFS-LP/observables/vfloat_std/value"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )
    lfs_lp.embedded[channel].v_floating_sigma.data_error_upper = np.array(
        [
            tcv_data["LFS-LP/observables/vfloat_std/error"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )

# IDS variable is filled, we write it now to the data entry
imas_entry.put(lfs_lp, 0)


##################### HFS-LP data
hfs_lp = imas.langmuir_probes()
hfs_lp.ids_properties.homogeneous_time = 1
hfs_lp.ids_properties.comment = tcv_data["HFS-LP"].diagnostic_name
hfs_lp.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
hfs_lp.time = np.array(
    [0.0]
)  # Time has no meaning for this dataset which is processed over several pulses and time slices
# midplane definition
hfs_lp.midplane.name = "dr_dz_zero_sep"
hfs_lp.midplane.index = 2
hfs_lp.midplane.description = "Midplane defined by the height of the outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen. equilibrium/time_slice/boundary_separatrix/dr_dz_zero_point/z"
# Different physical data have been gathered during different group of TCV pulses, therefore they are also measured at different locations. Typically : density is not measured at the same positions as jsat. For each set of measurement positions, we create a set of indices in the "embedded" AoS, since this one assumes that a given probe is located at given position
# Get number of channels for density/electron_temp/potential group
channels_n = len(tcv_data["HFS-LP/observables/density/value"])
# Get number of channels for current/current_std group
channels_c = len(tcv_data["HFS-LP/observables/current/value"])
# Get number of channels for jsat/jsat_std/jsat_skew/jsat_kurtosis group
channels_j = len(tcv_data["HFS-LP/observables/jsat/value"])
# Get number of channels for vfloat/vfloat_std group
channels_v = len(tcv_data["HFS-LP/observables/vfloat/value"])

hfs_lp.embedded.resize(channels_n + channels_c + channels_j + channels_v)

for channel in range(channels_n):
    print(channel)
    # Positions
    hfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [tcv_data["HFS-LP/observables/density/Rsep_omp"][channel] / 100.0]
    )
    # Electron density
    hfs_lp.embedded[channel].n_e.data = np.array(
        [tcv_data["HFS-LP/observables/density/value"][channel]]
    )
    hfs_lp.embedded[channel].n_e.data_error_upper = np.array(
        [tcv_data["HFS-LP/observables/density/error"][channel]]
    )
    # Electron temperature
    hfs_lp.embedded[channel].t_e.data = np.array(
        [tcv_data["HFS-LP/observables/electron_temp/value"][channel]]
    )
    hfs_lp.embedded[channel].t_e.data_error_upper = np.array(
        [tcv_data["HFS-LP/observables/electron_temp/error"][channel]]
    )
    # Plasma potential
    hfs_lp.embedded[channel].v_plasma.data = np.array(
        [tcv_data["HFS-LP/observables/potential/value"][channel]]
    )
    hfs_lp.embedded[channel].v_plasma.data_error_upper = np.array(
        [tcv_data["HFS-LP/observables/potential/error"][channel]]
    )

for channel in range(channels_n, channels_n + channels_c):
    print(channel)
    # Positions
    hfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [tcv_data["HFS-LP/observables/current/Rsep_omp"][channel - channels_n] / 100.0]
    )
    # Parallel current density
    hfs_lp.embedded[channel].j_i_parallel.data = np.array(
        [tcv_data["HFS-LP/observables/current/value"][channel - channels_n]]
    )
    hfs_lp.embedded[channel].j_i_parallel.data_error_upper = np.array(
        [tcv_data["HFS-LP/observables/current/error"][channel - channels_n]]
    )
    # Parallel current density standard deviation
    hfs_lp.embedded[channel].j_i_parallel_sigma.data = np.array(
        [tcv_data["HFS-LP/observables/current_std/value"][channel - channels_n]]
    )
    hfs_lp.embedded[channel].j_i_parallel_sigma.data_error_upper = np.array(
        [tcv_data["HFS-LP/observables/current_std/error"][channel - channels_n]]
    )

for channel in range(channels_n + channels_c, channels_n + channels_c + channels_j):
    print(channel)
    # Positions
    hfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [
            tcv_data["HFS-LP/observables/jsat/Rsep_omp"][
                channel - channels_n - channels_c
            ]
            / 100.0
        ]
    )
    # Ion saturation current density
    hfs_lp.embedded[channel].j_i_saturation.data = np.array(
        [tcv_data["HFS-LP/observables/jsat/value"][channel - channels_n - channels_c]]
    )
    hfs_lp.embedded[channel].j_i_saturation.data_error_upper = np.array(
        [tcv_data["HFS-LP/observables/jsat/error"][channel - channels_n - channels_c]]
    )
    # Ion saturation current density standard deviation
    hfs_lp.embedded[channel].j_i_saturation_sigma.data = np.array(
        [
            tcv_data["HFS-LP/observables/jsat_std/value"][
                channel - channels_n - channels_c
            ]
        ]
    )
    hfs_lp.embedded[channel].j_i_saturation_sigma.data_error_upper = np.array(
        [
            tcv_data["HFS-LP/observables/jsat_std/error"][
                channel - channels_n - channels_c
            ]
        ]
    )
    # Ion saturation current density skew
    hfs_lp.embedded[channel].j_i_saturation_skew.data = np.array(
        [
            tcv_data["HFS-LP/observables/jsat_skew/value"][
                channel - channels_n - channels_c
            ]
        ]
    )
    hfs_lp.embedded[channel].j_i_saturation_skew.data_error_upper = np.array(
        [
            tcv_data["HFS-LP/observables/jsat_skew/error"][
                channel - channels_n - channels_c
            ]
        ]
    )
    # Ion saturation current density kurtosis
    hfs_lp.embedded[channel].j_i_saturation_kurtosis.data = np.array(
        [
            tcv_data["HFS-LP/observables/jsat_kurtosis/value"][
                channel - channels_n - channels_c
            ]
        ]
    )
    hfs_lp.embedded[channel].j_i_saturation_kurtosis.data_error_upper = np.array(
        [
            tcv_data["HFS-LP/observables/jsat_kurtosis/error"][
                channel - channels_n - channels_c
            ]
        ]
    )

for channel in range(
    channels_n + channels_c + channels_j,
    channels_n + channels_c + channels_j + channels_v,
):
    print(channel)
    # Positions
    hfs_lp.embedded[channel].distance_separatrix_midplane.data = np.array(
        [
            tcv_data["HFS-LP/observables/vfloat/Rsep_omp"][
                channel - channels_n - channels_c - channels_j
            ]
            / 100.0
        ]
    )
    # Floating potential
    hfs_lp.embedded[channel].v_floating.data = np.array(
        [
            tcv_data["HFS-LP/observables/vfloat/value"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )
    hfs_lp.embedded[channel].v_floating.data_error_upper = np.array(
        [
            tcv_data["HFS-LP/observables/vfloat/error"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )
    # Floating potential standard deviation
    hfs_lp.embedded[channel].v_floating_sigma.data = np.array(
        [
            tcv_data["HFS-LP/observables/vfloat_std/value"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )
    hfs_lp.embedded[channel].v_floating_sigma.data_error_upper = np.array(
        [
            tcv_data["HFS-LP/observables/vfloat_std/error"][
                channel - channels_n - channels_c - channels_j
            ]
        ]
    )


# IDS variable is filled, we write it now to the data entry
imas_entry.put(hfs_lp, 1)

##################### LFS_IR data
lfs_ir = imas.camera_ir()
lfs_ir.ids_properties.homogeneous_time = 1
lfs_ir.ids_properties.comment = tcv_data["LFS-IR"].diagnostic_name
lfs_ir.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
lfs_ir.time = np.array(
    [0.0]
)  # Time has no meaning for this dataset which is processed over several pulses and time slices
# midplane definition
lfs_ir.midplane.name = "dr_dz_zero_sep"
lfs_ir.midplane.index = 2
lfs_ir.midplane.description = "Midplane defined by the height of the outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen. equilibrium/time_slice/boundary_separatrix/dr_dz_zero_point/z"

lfs_ir.frame_analysis.resize(1)  # 1 time slice
# Position
lfs_ir.frame_analysis[0].distance_separatrix_midplane = np.array(
    tcv_data["LFS-IR/observables/q_parallel/Rsep_omp"][:] / 100.0
)
lfs_ir.frame_analysis[0].power_flux_parallel = np.array(
    tcv_data["LFS-IR/observables/q_parallel/value"]
)
lfs_ir.frame_analysis[0].power_flux_parallel_error_upper = np.array(
    tcv_data["LFS-IR/observables/q_parallel/error"]
)
# IDS variable is filled, we write it now to the data entry
imas_entry.put(lfs_ir, 0)

##################### FHRP data
fhrp = imas.langmuir_probes()
fhrp.ids_properties.homogeneous_time = 1
fhrp.ids_properties.comment = tcv_data["FHRP"].diagnostic_name
fhrp.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
fhrp.time = np.array(
    [0.0]
)  # Time has no meaning for this dataset which is processed over several pulses and time slices
# midplane definition
fhrp.midplane.name = "dr_dz_zero_sep"
fhrp.midplane.index = 2
fhrp.midplane.description = "Midplane defined by the height of the outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen. equilibrium/time_slice/boundary_separatrix/dr_dz_zero_point/z"
# Different physical data have been gathered during different group of TCV pulses, therefore they are also measured at different locations. Typically : density is not measured at the same positions as jsat. For each set of measurement positions, we create a set of indices in the "reciprocating" AoS, since this one assumes that a given probe is located at given position
# Get number of channels for density/electron_temp/potential group
channels_n = len(tcv_data["FHRP/observables/density/value"])
# Get number of channels for jsat/jsat_std/jsat_skew/jsat_kurtosis/vfloat/vfloat_std/mach_number group
channels_j = len(tcv_data["FHRP/observables/jsat/value"])

fhrp.reciprocating.resize(channels_n + channels_j)

for channel in range(channels_n):
    print(channel)
    fhrp.reciprocating[channel].plunge.resize(1)
    fhrp.reciprocating[channel].plunge[0].collector.resize(1)
    # Positions
    fhrp.reciprocating[channel].plunge[0].distance_separatrix_midplane.data = np.array(
        [tcv_data["FHRP/observables/density/Rsep_omp"][channel] / 100.0]
    )
    # Electron density
    fhrp.reciprocating[channel].plunge[0].n_e.data = np.array(
        [tcv_data["FHRP/observables/density/value"][channel]]
    )
    fhrp.reciprocating[channel].plunge[0].n_e.data_error_upper = np.array(
        [tcv_data["FHRP/observables/density/error"][channel]]
    )
    # Electron temperature
    fhrp.reciprocating[channel].plunge[0].collector[0].t_e.data = np.array(
        [tcv_data["FHRP/observables/electron_temp/value"][channel]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[0].t_e.data_error_upper = np.array(
        [tcv_data["FHRP/observables/electron_temp/error"][channel]]
    )
    # Plasma potential
    fhrp.reciprocating[channel].plunge[0].v_plasma.data = np.array(
        [tcv_data["FHRP/observables/potential/value"][channel]]
    )
    fhrp.reciprocating[channel].plunge[0].v_plasma.data_error_upper = np.array(
        [tcv_data["FHRP/observables/potential/error"][channel]]
    )


for channel in range(channels_n, channels_n + channels_j):
    print(channel)
    fhrp.reciprocating[channel].plunge.resize(1)
    fhrp.reciprocating[channel].plunge[0].collector.resize(1)
    # Positions
    fhrp.reciprocating[channel].plunge[0].distance_separatrix_midplane.data = np.array(
        [tcv_data["FHRP/observables/jsat/Rsep_omp"][channel - channels_n] / 100.0]
    )
    # Ion saturation current density
    fhrp.reciprocating[channel].plunge[0].collector[0].j_i_saturation.data = np.array(
        [tcv_data["FHRP/observables/jsat/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_saturation.data_error_upper = np.array(
        [tcv_data["FHRP/observables/jsat/error"][channel - channels_n]]
    )
    # Ion saturation current density standard deviation
    fhrp.reciprocating[channel].plunge[0].collector[0].j_i_sigma.data = np.array(
        [tcv_data["FHRP/observables/jsat_std/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_sigma.data_error_upper = np.array(
        [tcv_data["FHRP/observables/jsat_std/error"][channel - channels_n]]
    )
    # Ion saturation current density skew
    fhrp.reciprocating[channel].plunge[0].collector[0].j_i_skew.data = np.array(
        [tcv_data["FHRP/observables/jsat_skew/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_skew.data_error_upper = np.array(
        [tcv_data["FHRP/observables/jsat_skew/error"][channel - channels_n]]
    )
    # Ion saturation current density kurtosis
    fhrp.reciprocating[channel].plunge[0].collector[0].j_i_kurtosis.data = np.array(
        [tcv_data["FHRP/observables/jsat_kurtosis/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_kurtosis.data_error_upper = np.array(
        [tcv_data["FHRP/observables/jsat_kurtosis/error"][channel - channels_n]]
    )
    # Floating potential
    fhrp.reciprocating[channel].plunge[0].collector[0].v_floating.data = np.array(
        [tcv_data["FHRP/observables/vfloat/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[
        0
    ].v_floating.data_error_upper = np.array(
        [tcv_data["FHRP/observables/vfloat/error"][channel - channels_n]]
    )
    # Floating potential standard deviation
    fhrp.reciprocating[channel].plunge[0].collector[0].v_floating_sigma.data = np.array(
        [tcv_data["FHRP/observables/vfloat_std/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[0].collector[
        0
    ].v_floating_sigma.data_error_upper = np.array(
        [tcv_data["FHRP/observables/vfloat_std/error"][channel - channels_n]]
    )
    # Mach number
    fhrp.reciprocating[channel].plunge[0].mach_number_parallel.data = np.array(
        [tcv_data["FHRP/observables/mach_number/value"][channel - channels_n]]
    )
    fhrp.reciprocating[channel].plunge[
        0
    ].mach_number_parallel.data_error_upper = np.array(
        [tcv_data["FHRP/observables/mach_number/error"][channel - channels_n]]
    )

# IDS variable is filled, we write it now to the data entry
imas_entry.put(fhrp, 2)

##################### RDPA data
rdpa = imas.langmuir_probes()
rdpa.ids_properties.homogeneous_time = 1
rdpa.ids_properties.comment = tcv_data["RDPA"].diagnostic_name
rdpa.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
rdpa.time = np.array(
    [0.0]
)  # Time has no meaning for this dataset which is processed over several pulses and time slices
# midplane definition
rdpa.midplane.name = "dr_dz_zero_sep"
rdpa.midplane.index = 2
rdpa.midplane.description = "Midplane defined by the height of the outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen. equilibrium/time_slice/boundary_separatrix/dr_dz_zero_point/z"
# Different physical data have been gathered during different group of TCV pulses, therefore they are also measured at different locations. Typically : density is not measured at the same positions as jsat. For each set of measurement positions, we create a set of indices in the "reciprocating" AoS, since this one assumes that a given probe is located at given position
# Get number of channels for density/electron_temp/potential/mach_number group
channels_n = len(tcv_data["RDPA/observables/density/value"])
# Get number of channels for jsat/jsat_std/jsat_skew/jsat_kurtosis/vfloat/vfloat_std/mach_number group
channels_j = len(tcv_data["RDPA/observables/jsat/value"])
# Get number of channels for vfloat/vfloat_std group
channels_v = len(tcv_data["RDPA/observables/vfloat/value"])

rdpa.reciprocating.resize(channels_n + channels_j + channels_v)

for channel in range(channels_n):
    print(channel)
    rdpa.reciprocating[channel].plunge.resize(1)
    rdpa.reciprocating[channel].plunge[0].collector.resize(1)
    # Positions
    rdpa.reciprocating[channel].plunge[0].distance_separatrix_midplane.data = np.array(
        [tcv_data["RDPA/observables/density/Rsep_omp"][channel] / 100.0]
    )
    rdpa.reciprocating[channel].plunge[0].distance_x_point_z.data = np.array(
        [tcv_data["RDPA/observables/density/Zx"][channel]]
    )
    # Electron density
    rdpa.reciprocating[channel].plunge[0].n_e.data = np.array(
        [tcv_data["RDPA/observables/density/value"][channel]]
    )
    rdpa.reciprocating[channel].plunge[0].n_e.data_error_upper = np.array(
        [tcv_data["RDPA/observables/density/error"][channel]]
    )
    # Electron temperature
    rdpa.reciprocating[channel].plunge[0].collector[0].t_e.data = np.array(
        [tcv_data["RDPA/observables/electron_temp/value"][channel]]
    )
    rdpa.reciprocating[channel].plunge[0].collector[0].t_e.data_error_upper = np.array(
        [tcv_data["RDPA/observables/electron_temp/error"][channel]]
    )
    # Plasma potential
    rdpa.reciprocating[channel].plunge[0].v_plasma.data = np.array(
        [tcv_data["RDPA/observables/potential/value"][channel]]
    )
    rdpa.reciprocating[channel].plunge[0].v_plasma.data_error_upper = np.array(
        [tcv_data["RDPA/observables/potential/error"][channel]]
    )
    # Mach number
    rdpa.reciprocating[channel].plunge[0].mach_number_parallel.data = np.array(
        [tcv_data["RDPA/observables/mach_number/value"][channel]]
    )
    rdpa.reciprocating[channel].plunge[
        0
    ].mach_number_parallel.data_error_upper = np.array(
        [tcv_data["RDPA/observables/mach_number/error"][channel]]
    )


for channel in range(channels_n, channels_n + channels_j):
    print(channel)
    rdpa.reciprocating[channel].plunge.resize(1)
    rdpa.reciprocating[channel].plunge[0].collector.resize(1)
    # Positions
    rdpa.reciprocating[channel].plunge[0].distance_separatrix_midplane.data = np.array(
        [tcv_data["RDPA/observables/jsat/Rsep_omp"][channel - channels_n] / 100.0]
    )
    rdpa.reciprocating[channel].plunge[0].distance_x_point_z.data = np.array(
        [tcv_data["RDPA/observables/jsat/Zx"][channel - channels_n]]
    )
    # Ion saturation current density
    rdpa.reciprocating[channel].plunge[0].collector[0].j_i_saturation.data = np.array(
        [tcv_data["RDPA/observables/jsat/value"][channel - channels_n]]
    )
    rdpa.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_saturation.data_error_upper = np.array(
        [tcv_data["RDPA/observables/jsat/error"][channel - channels_n]]
    )
    # Ion saturation current density standard deviation
    rdpa.reciprocating[channel].plunge[0].collector[0].j_i_sigma.data = np.array(
        [tcv_data["RDPA/observables/jsat_std/value"][channel - channels_n]]
    )
    rdpa.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_sigma.data_error_upper = np.array(
        [tcv_data["RDPA/observables/jsat_std/error"][channel - channels_n]]
    )
    # Ion saturation current density skew
    rdpa.reciprocating[channel].plunge[0].collector[0].j_i_skew.data = np.array(
        [tcv_data["RDPA/observables/jsat_skew/value"][channel - channels_n]]
    )
    rdpa.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_skew.data_error_upper = np.array(
        [tcv_data["RDPA/observables/jsat_skew/error"][channel - channels_n]]
    )
    # Ion saturation current density kurtosis
    rdpa.reciprocating[channel].plunge[0].collector[0].j_i_kurtosis.data = np.array(
        [tcv_data["RDPA/observables/jsat_kurtosis/value"][channel - channels_n]]
    )
    rdpa.reciprocating[channel].plunge[0].collector[
        0
    ].j_i_kurtosis.data_error_upper = np.array(
        [tcv_data["RDPA/observables/jsat_kurtosis/error"][channel - channels_n]]
    )

for channel in range(channels_n + channels_j, channels_n + channels_j + channels_v):
    print(channel)
    rdpa.reciprocating[channel].plunge.resize(1)
    rdpa.reciprocating[channel].plunge[0].collector.resize(1)
    # Positions
    rdpa.reciprocating[channel].plunge[0].distance_separatrix_midplane.data = np.array(
        [
            tcv_data["RDPA/observables/vfloat/Rsep_omp"][
                channel - channels_n - channels_j
            ]
            / 100.0
        ]
    )
    rdpa.reciprocating[channel].plunge[0].distance_x_point_z.data = np.array(
        [tcv_data["RDPA/observables/vfloat/Zx"][channel - channels_n - channels_j]]
    )
    # Floating potential
    rdpa.reciprocating[channel].plunge[0].collector[0].v_floating.data = np.array(
        [tcv_data["RDPA/observables/vfloat/value"][channel - channels_n - channels_j]]
    )
    rdpa.reciprocating[channel].plunge[0].collector[
        0
    ].v_floating.data_error_upper = np.array(
        [tcv_data["RDPA/observables/vfloat/error"][channel - channels_n - channels_j]]
    )
    # Floating potential standard deviation
    rdpa.reciprocating[channel].plunge[0].collector[0].v_floating_sigma.data = np.array(
        [
            tcv_data["RDPA/observables/vfloat_std/value"][
                channel - channels_n - channels_j
            ]
        ]
    )
    rdpa.reciprocating[channel].plunge[0].collector[
        0
    ].v_floating_sigma.data_error_upper = np.array(
        [
            tcv_data["RDPA/observables/vfloat_std/error"][
                channel - channels_n - channels_j
            ]
        ]
    )

# IDS variable is filled, we write it now to the data entry
imas_entry.put(rdpa, 3)

##################### TS data
ts = imas.thomson_scattering()
ts.ids_properties.homogeneous_time = 1
ts.ids_properties.comment = tcv_data["TS"].diagnostic_name
ts.ids_properties.provider = "F. Imbeaux (for the IMAS conversion)"
ts.time = np.array(
    [0.0]
)  # Time has no meaning for this dataset which is processed over several pulses and time slices
# midplane definition
ts.midplane.name = "dr_dz_zero_sep"
ts.midplane.index = 2
ts.midplane.description = "Midplane defined by the height of the outboard point on the separatrix on which dr/dz = 0 (local maximum of the major radius of the separatrix). In case of multiple local maxima, the closest one from z=z_magnetic_axis is chosen. equilibrium/time_slice/boundary_separatrix/dr_dz_zero_point/z"

# Get number of channels for density/electron_temp group
channels_n = len(tcv_data["TS/observables/density/value"])

ts.channel.resize(channels_n)

for channel in range(channels_n):
    print(channel)
    # Positions
    ts.channel[channel].distance_separatrix_midplane.data = np.array(
        [tcv_data["RDPA/observables/density/Rsep_omp"][channel] / 100.0]
    )
    # Electron density
    ts.channel[channel].n_e.data = np.array(
        [tcv_data["RDPA/observables/density/value"][channel]]
    )
    ts.channel[channel].n_e.data_error_upper = np.array(
        [tcv_data["RDPA/observables/density/error"][channel]]
    )
    # Electron temperature
    ts.channel[channel].t_e.data = np.array(
        [tcv_data["RDPA/observables/electron_temp/value"][channel]]
    )
    ts.channel[channel].t_e.data_error_upper = np.array(
        [tcv_data["RDPA/observables/electron_temp/error"][channel]]
    )

# IDS variable is filled, we write it now to the data entry
imas_entry.put(ts, 0)

imas_entry.close()
