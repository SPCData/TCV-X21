#!/usr/bin/env python3
#
# This script extracts data from a Hermes-3 simulation output,
# saving a summary into a pickle file. These pickle files can be
# combined for analysis of longer timeseries.
# 
# The output of this script is input to convert_to_tcvx21

import numpy as np
from boutdata import collect
from boutdata.shiftz import shiftz
from boututils.datafile import DataFile
import pickle

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root

import matplotlib.pyplot as plt

TWOPI = 2*np.pi
qe = 1.602e-19
AA_me = 1.0 / 1836  # Electron mass divided by proton mass

AA = 2  # Atomic species
electron_adiabatic = 5./3
ion_adiabatic = 5./3
Ge = 0.0  # Secondary electron emission

zperiod = 5

def extract_data(path, gridfilepath, ymid=18):
    """
    Read data from a Hermes-3/BOUT++ output directory.
    Returned as a dictionary.

    Note: Here we should use xHermes, but the grid file does not
    include the Y guard cell values. xHermes/xBOUT therefore fails
    to load the dataset due to array size mismatches if Y boundaries
    are requested.

    Since extrapolate_y = false in this case, the zShift angle is
    constant into the boundary. Hence no mapping to field-aligned
    coordinates is needed to reconstruct sheath entrance values.

    NOTE: The code below is specific to this TCV-X21 case!
    """

    # Many diagnostics are mapped in flux space to R at midplane
    with DataFile(gridfilepath) as grid:
        psixy = grid["psixy"]
        Rxy = grid["Rxy"]
        Zxy = grid["Zxy"]
        ixsep = grid["ixseps1"]
        jyseps1 = grid["jyseps1_1"]
        jyseps2 = grid["jyseps2_2"]
        zShift = grid["zShift"]

    Zx = 0.25*(Zxy[ixsep, jyseps1] + Zxy[ixsep, jyseps1 + 1] + Zxy[ixsep, jyseps2] + Zxy[ixsep, jyseps2 + 1])

    psi_mid = psixy[:, ymid]
    R_mid = Rxy[:, ymid]
    Rsep = 0.5 * (R_mid[ixsep - 1] + R_mid[ixsep])

    from scipy.interpolate import interp1d

    R_u = interp1d(
        psi_mid, R_mid - Rsep, fill_value="extrapolate"
    )  # Linearly interpolate R as a function of psi
    
    # The TS and RDPA diagnostics require interpolation in 2D
    # Interpolation must be done in flux space due to sparse grid in poloidal angle

    def make_interpolator(dataxy, method='slinear'):
        """
        Create a 2D interpolator that takes into account branch cuts
        """
        nx, ny = dataxy.shape
        d = dataxy[:, :(jyseps1+2)]
        d[:ixsep, -1] = dataxy[:ixsep, jyseps2+1] # Connect PF region        
        inner = RegularGridInterpolator((np.arange(nx), np.arange(jyseps1+2)),
                                        d,
                                        method=method)
        d = dataxy[:, jyseps1:jyseps2+2]
        d[:ixsep, 0] = dataxy[:ixsep, jyseps2] # Connect core
        d[:ixsep, -1] = dataxy[:ixsep, jyseps1+1] # Connect core
        core = RegularGridInterpolator((np.arange(nx), np.arange(jyseps1, jyseps2+2)),
                                       d,
                                       method=method)
        d = dataxy[:, jyseps2:ny]
        d[:ixsep, 0] = dataxy[:ixsep, jyseps1] # Connect PF region
        outer = RegularGridInterpolator((np.arange(nx), np.arange(jyseps2, ny)),
                                        d,
                                        method=method)
        def data(x, y):
            x = np.clip(x, 0, nx-1)
            y = np.clip(y, 0, ny-1)
            x0 = int(x + 0.5)
            y0 = int(y + 0.5)
            if y0 <= jyseps1:
                return inner((x, y))
            if y0 > jyseps2:
                return outer((x, y))
            return core((x, y))
        return data

    R = make_interpolator(Rxy)
    Z = make_interpolator(Zxy)
    psi = make_interpolator(psixy, method='slinear')

    def RZfunc(xy, R0, Z0):
        return [R(xy[0], xy[1]) - R0, Z(xy[0], xy[1]) - Z0]

    # Find the (x,y) coordinates of each TS point
    R_TS = 0.9 # m
    Z_TS = np.linspace(-0.33, -0.69, num=100) # m
    X_TS = np.ndarray(len(Z_TS))
    Y_TS = np.ndarray(len(Z_TS))
    Ru_TS = np.ndarray(len(Z_TS))

    xy0 = [ 4.71012195, 23.44114604]
    for i, Zpos in enumerate(Z_TS):
        sol = root(RZfunc, xy0, args=(R_TS, Zpos), jac=False, method='krylov')
        if not sol.success:
            raise ValueError("Couldn't find interpolation")
        X_TS[i] = sol.x[0]
        Y_TS[i] = sol.x[1]
        Ru_TS[i] = R_u(psi(sol.x[0], sol.x[1]))
        xy0 = sol.x # Start next point

    # Find coordinates for RDPA diagnostic
    # Data is a regular grid in Ru (flux) and 0 > Zx > -0.32
    Rs = np.linspace(0.77, 0.9, num=25)
    Zs = np.linspace(Zx, Zx-0.32, num=25)
    R_RDPA = []
    Z_RDPA = []
    for i in range(len(Zs)):
        R_RDPA += list(Rs[::(-1)**i]) # Raster scan, so consecutive points are close together
        Z_RDPA += [Zs[i]] * len(Rs)
    R_RDPA = np.array(R_RDPA)
    Z_RDPA = np.array(Z_RDPA)
    Z_Zx_RDPA = Z_RDPA - Zx
    X_RDPA = np.ndarray(len(R_RDPA))
    Y_RDPA = np.ndarray(len(Z_RDPA))
    Ru_RDPA = np.ndarray(len(Z_RDPA))

    xy0 = [ixsep, jyseps2+1]
    for i in range(len(R_RDPA)):
        sol = root(RZfunc, xy0, args=(R_RDPA[i], Z_RDPA[i]), jac=False, method='krylov')
        if not sol.success:
            raise ValueError("Couldn't find interpolation: ({}, {})".format(R_RDPA[i], Z_RDPA[i]))
        X_RDPA[i] = sol.x[0]
        Y_RDPA[i] = sol.x[1]
        Ru_RDPA[i] = R_u(psi(sol.x[0], sol.x[1]))
        xy0 = sol.x # Start next point

    Nnorm = collect("Nnorm", path=path)
    Tnorm = collect("Tnorm", path=path)
    wci = collect("Omega_ci", path=path)
    Cs0 = collect("Cs0", path=path)
    time = collect("t_array", path=path) / wci  # Seconds
    run_id = collect("run_id", path=path)

    # Dictionary to be populated by the add_var function
    result = {}

    def hfs(data_txyz):
        return 0.5 * (data_txyz[:, :, 1, :] + data_txyz[:, :, 2, :])
    def lfs(data_txyz):
        return 0.5 * (data_txyz[:, :, -2, :] + data_txyz[:, :, -3, :])

    def add_location(location, name, units, Rx, data_txz, Zx=None):
        assert len(Rx.shape) == 1
        assert len(data_txz.shape) == 3
        assert len(Rx) == data_txz.shape[1]

        result[name][location] = {
            "name": name,
            "units": units,
            "nt": data_txz.shape[0],
            "tmin": time[0],
            "tmax": time[-1],
            "duration": time[-1] - time[0],
            "nz": data_txz.shape[-1],
            "run_ids": [run_id],
            "paths": [path],
            "gridfilepath": gridfilepath,

            # Calculate moments about the origin.  These can be
            # combined from multiple datasets and then converted
            # to central moments (std, skew, kurt).
            "mean": np.mean(data_txz, axis=(0, -1)),  # Average in time and Z
            "mean2": np.mean(data_txz**2, axis=(0, -1)),
            "mean3": np.mean(data_txz**3, axis=(0, -1)),
            "mean4": np.mean(data_txz**4, axis=(0, -1)),
            "std": np.std(data_txz, axis=(0, -1)),
            "Ru": Rx * 100.0,  # Major radius in cm
            "Ru_units": "cm",
        }
        if Zx is not None:
            result[name][location]['Zx'] = Zx
            result[name][location]['Zx_units'] = 'm'

    def add_var(name, units, data_txyz):
        assert len(data_txyz.shape) == 4
        assert data_txyz.shape[1] == psixy.shape[0]  # X axis
        assert data_txyz.shape[2] == psixy.shape[1] + 4  # Y axis.
        # Data includes 4 Y guards that are not in the grid file

        result[name] = {}

        # Outboard midplane. data includes 2 Y guard cells
        add_location("omp", name, units, R_mid - Rsep, data_txyz[:, :, ymid + 2, :])

        # Interpolate at high field side, mapping to outboard midplane R
        add_location(
            "hfs",
            name,
            units,
            R_u(psixy[:, 0]),
            hfs(data_txyz),
        )

        # Interpolate at low field side
        add_location(
            "lfs",
            name,
            units,
            R_u(psixy[:, -1]),
            lfs(data_txyz),
        )

    Ne = collect("Ne", path=path, yguards=True)
    add_var("Ne", "1/m^3", Ne * Nnorm)

    Ne_floor = np.clip(Ne, 1e-5, None)

    Pe = collect("Pe", path=path, yguards=True)
    add_var("Pe", "Pa", Pe * Nnorm * Tnorm * qe)

    Te = Pe / Ne_floor
    add_var("Te", "eV", Te * Tnorm)

    Pi = collect("Pi", path=path, yguards=True)
    add_var("Pi", "Pa", Pi * Nnorm * Tnorm * qe)

    Ti = Pi / Ne_floor
    add_var("Ti", "eV", Ti * Tnorm)

    phi = collect("phi", path=path, yguards=True)
    add_var("phi", "V", phi * Tnorm)

    Vfl = phi - 3 * Te
    add_var("Vfl", "V", Vfl * Tnorm)

    # Note: NVi and NVe contain mass factors
    NVi = collect("NVi", path=path, yguards=True)
    Vi = NVi / (AA * Ne_floor)
    add_var("Vi", "m/s",  Vi * Cs0)

    Cs = np.sqrt((ion_adiabatic * Ti + Te)/AA)
    Mpar = Vi / Cs # Mach number
    add_var("Mpar", "", Mpar)

    NVe = collect("NVe", path=path, yguards=True)
    Jpar = NVi / AA - NVe / AA_me
    add_var("Jpar", "A/m^2", Jpar * qe * Nnorm * Cs0)

    # Ion saturation current
    Jsat = qe * np.sqrt((Te + ion_adiabatic*Ti) * (Tnorm * qe / (1.67e-27 * AA))) * Ne * Nnorm
    add_var("Jsat", "A/m^2", Jsat)

    # Heat flux at inner and outer target
    for name in ["qpar", "qpar_e", "qpar_i",
                 "gamma_e", "gamma_i",
                 "ion_parallel_flux", "electron_parallel_flux"]:
        result[name] = {}

    def heat_flux(targ, location, Rx):
        """
        Reproducing sheath_boundary Hermes-3 component
        """
        phisheath = np.clip(targ(phi), 0.0, None)
        tesheath = targ(Te)
        tisheath = targ(Ti)
        nesheath = targ(Ne)

        nvesheath = np.abs(targ(NVe)) / AA_me
        nvisheath = np.abs(targ(NVi)) / AA

        gamma_e = np.clip(2/(1 - Ge) + phisheath / np.clip(tesheath, 1e-5, None), 0.0, None)
        vesheath = np.sqrt(tesheath / (TWOPI * AA_me)) * (1 - Ge) * np.exp(-phisheath / tesheath)

        # Electron heat flux
        Qe = gamma_e * tesheath * nvesheath

        C_i_sq = np.clip((ion_adiabatic * tisheath + tesheath) / AA, 0, 100)
        gamma_i = 2.5 + 0.5 * AA * C_i_sq / tisheath
        visheath = np.sqrt(C_i_sq)
        Qi = gamma_i * tisheath * nvisheath

        # Convert to SI units
        qnorm = qe * Tnorm * Nnorm * Cs0

        add_location(location, "gamma_e", "", Rx, gamma_e)
        add_location(location, "gamma_i", "", Rx, gamma_i)

        add_location(location, "ion_parallel_flux", "m^-2", Rx, nvisheath * Nnorm * Cs0)
        add_location(location, "electron_parallel_flux", "m^-2", Rx, nvesheath * Nnorm * Cs0)

        add_location(location, "qpar_e", "W/m^2", Rx, Qe * qnorm)
        add_location(location, "qpar_i", "W/m^2", Rx, Qi * qnorm)
        add_location(location, "qpar", "W/m^2", Rx, (Qe + Qi) * qnorm)

    heat_flux(hfs, "hfs", R_u(psixy[:, 0]))
    heat_flux(lfs, "lfs", R_u(psixy[:, -1]))

    # Shift into field-aligned coordinates for TS and RDPA interpolation
    Ne = shiftz(Ne[:,:,2:-2,:], -zShift, zperiod=zperiod)
    Te = shiftz(Te[:,:,2:-2,:], -zShift, zperiod=zperiod)
    Ti = shiftz(Ti[:,:,2:-2,:], -zShift, zperiod=zperiod)
    phi = shiftz(phi[:,:,2:-2,:], -zShift, zperiod=zperiod)
    Vfl = shiftz(Vfl[:,:,2:-2,:], -zShift, zperiod=zperiod)
    Mpar = shiftz(Mpar[:,:,2:-2,:], -zShift, zperiod=zperiod)
    Jsat = shiftz(Jsat[:,:,2:-2,:], -zShift, zperiod=zperiod)

    nt = Ne.shape[0]
    nz = Ne.shape[-1]

    # Interpolate onto TS and RDPA locations
    def interpolate_TS_RDPA(var):
        var_TS = np.ndarray((nt, len(Ru_TS), nz))
        var_RDPA = np.ndarray((nt, len(Ru_RDPA), nz))
        for t in range(nt):
            for z in range(nz):
                print(f"\rt = {t}, z = {z}", end=' ')
                # Create interpolator in 2D
                n = make_interpolator(var[t, :, :, z])
                for i in range(len(Ru_TS)):
                    var_TS[t,i,z] = n(X_TS[i], Y_TS[i])
                for i in range(len(Ru_RDPA)):
                    var_RDPA[t,i,z] = n(X_RDPA[i], Y_RDPA[i])
        return var_TS, var_RDPA

    print("Interpolating Ne")
    Ne_TS, Ne_RDPA = interpolate_TS_RDPA(Ne)
    add_location("TS", "Ne", "1/m^3", Ru_TS, Ne_TS * Nnorm)
    add_location("RDPA", "Ne", "1/m^3", Ru_RDPA, Ne_RDPA * Nnorm, Zx=Z_Zx_RDPA)

    print("\nInterpolating Te")
    Te_TS, Te_RDPA = interpolate_TS_RDPA(Te)
    add_location("TS", "Te", "eV", Ru_TS, Te_TS * Tnorm)
    add_location("RDPA", "Te", "eV", Ru_RDPA, Te_RDPA * Tnorm, Zx=Z_Zx_RDPA)

    print("\nInterpolating Ti")
    Ti_TS, Ti_RDPA = interpolate_TS_RDPA(Ti)
    add_location("TS", "Ti", "eV", Ru_TS, Ti_TS * Tnorm)
    add_location("RDPA", "Ti", "eV", Ru_RDPA, Ti_RDPA * Tnorm, Zx=Z_Zx_RDPA)

    print("\nInterpolating phi")
    _, phi_RDPA = interpolate_TS_RDPA(phi)
    add_location("RDPA", "phi", "V", Ru_RDPA, phi_RDPA * Tnorm, Zx=Z_Zx_RDPA)

    print("\nInterpolating Vfl")
    _, Vfl_RDPA = interpolate_TS_RDPA(Vfl)
    add_location("RDPA", "Vfl", "V", Ru_RDPA, Vfl_RDPA * Tnorm, Zx=Z_Zx_RDPA)

    print("\nInterpolating Mpar")
    _, Mpar_RDPA = interpolate_TS_RDPA(Mpar)
    add_location("RDPA", "Mpar", "", Ru_RDPA, Mpar_RDPA, Zx=Z_Zx_RDPA)

    print("\nInterpolating Jsat")
    _, Jsat_RDPA = interpolate_TS_RDPA(Jsat)
    add_location("RDPA", "Jsat", "A/m^2", Ru_RDPA, Jsat_RDPA, Zx=Z_Zx_RDPA)

    return result


def combine_data(dataset1, dataset2):
    """
    Combine two datasets together. Returns a new dataset.
    """
    result = {}
    for name in dataset1.keys():
        result[name] = {}
        for location in dataset1[name].keys():
            data1 = dataset1[name][location]
            data2 = dataset2[name][location]

            # Check that the datasets are compatible
            assert data1["name"] == data2["name"]
            assert data1["units"] == data2["units"]
            assert data1["nz"] == data2["nz"]
            assert data1["gridfilepath"] == data2["gridfilepath"]
            assert data1["Ru_units"] == data2["Ru_units"]

            # Weighting factors for combining means
            nt_tot = data1["nt"] + data2["nt"]
            w1 = data1["nt"] / nt_tot
            w2 = data2["nt"] / nt_tot
            mean = data1["mean"] * w1 + data2["mean"] * w2
            mean2 = data1["mean2"] * w1 + data2["mean2"] * w2
            mean3 = data1["mean3"] * w1 + data2["mean3"] * w2
            mean4 = data1["mean4"] * w1 + data2["mean4"] * w2
            # Re-calculate standard deviation
            std = np.sqrt(mean2 - mean**2)

            result[name][location] = {
                "name": data1["name"],
                "units": data1["units"],
                "nt": nt_tot,
                "tmin": min([data1["tmin"], data2["tmin"]]),
                "tmax": max([data1["tmax"], data2["tmax"]]),
                "duration": data1["duration"] + data2["duration"],
                "nz": data1["nz"],
                "run_ids": data1["run_ids"] + data2["run_ids"],
                "paths": data1["paths"] + data2["paths"],
                "gridfilepath": data1["gridfilepath"],
                "mean": mean,
                "mean2": mean2,
                "mean3": mean3,
                "mean4": mean4,
                "std": std,
                "Ru": data1["Ru"],
                "Ru_units": data1["Ru_units"],
            }
            # 2D datasets (RDPA) have a Z location
            if "Zx" in data1:
                result[name][location]["Zx"] = data1["Zx"]
                result[name][location]["Zx_units"] = data1["Zx_units"]

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gather Hermes-3 data from one or more directories"
    )

    parser.add_argument(
        "paths", type=str, nargs="+", help="Paths containing datasets to concatenate"
    )

    parser.add_argument(
        "-o", "--output", default="data.pickle", help="Pickle file to write to"
    )

    parser.add_argument(
        "-g", "--grid", default="bout.grd.nc", type=str, help="Grid file"
    )

    args = parser.parse_args()

    print(f"Got {len(args.paths)} files: {args.paths}")
    print(f"Outputting to '{args.output}'")

    data = extract_data(args.paths[0], args.grid, ymid=18)
    for path in args.paths[1:]:
        data2 = extract_data(path, args.grid, ymid=18)
        data = combine_data(data, data2)

    with open(args.output, "wb") as f:
        pickle.dump(data, f)
