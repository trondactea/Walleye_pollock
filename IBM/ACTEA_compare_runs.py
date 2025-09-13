import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
import trajan as ta


def read_and_compare_trajectories():
    ds1 = xr.open_dataset('output_walleye_pollock_eggs/walleye_pollock_eggs_drift_constant_egg_density.nc', decode_coords=False)
    # Requirement that status>=0 is needed since non-valid points are not masked in OpenDrift output
    ds1 = ds1.where(ds1.status >= 0)  # only active particles

    kw={'kwargs': {"fast":True}}

    ds1.traj.plot(color='tab:red', alpha=0.1, land='mask')
    dmean = ds1.mean('trajectory', skipna=True)
    dmean.traj.plot(color='tab:red', linewidth=2)
    ax = plt.gca()
    ax.set_title('Walleye pollock eggs dynamic density')
    plt.tight_layout()
    plt.savefig("walleye_pollock_eggs/Figures/walleye_pollock_eggs_drift_constant_egg_density.png", dpi=300)

    ds2 = xr.open_dataset('output_walleye_pollock_eggs/walleye_pollock_eggs_drift_dynamic_egg_density_light.nc', decode_coords=False)
    # Requirement that status>=0 is needed since non-valid points are not masked in OpenDrift output
    ds2 = ds2.where(ds2.status >= 0)  # only active particles
    ds2.traj.plot(color='tab:blue', alpha=0.1, land='mask')
    dmean = ds2.mean('trajectory', skipna=True)
    dmean.traj.plot(color='tab:blue', linewidth=2, ax=ax)
    plt.tight_layout()
    plt.savefig("walleye_pollock_eggs/Figures/walleye_pollock_eggs_drift_dynamic_egg_density_light.png", dpi=300)

read_and_compare_trajectories()