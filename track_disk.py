#!/usr/bin/env python3
import pandas as pd
import numpy as np
import string
from random import randint
import matplotlib.pyplot as plt

from src.combine import CombinedFile
from src.identify import ParticleMap

# use seaborn colorblind palette
plt.style.use('seaborn-colorblind')

runs = [
    {
        "name": "mars_citron_1vesc_b073_stewart_undiff",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
        "num_processes": 400,
    }
]

# define the iteration parameters
start_iteration = 0
end_iteration = 500
increment = 5

# define the planet parameters
mass_planet = 6.39e23
equatorial_radius = 3390e3

# define some misc stuff
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
axes = ['times', 'disk_entropy', 'disk_temperature', 'disk_vmf', 'disk_mass', 'disk_angular_momentum',
        'disk_impactor_mass_fraction']
ylabels = ["Avg. Disk Entropy (J/kg/K)", "Avg. Disk Temperature (K)", "Disk VMF (%)", "Disk Mass (kg)",
             r"Disk Angular Momentum ($L_{\rm MM}$)", "Disk Impactor Mass Fraction (%)"]

# collect disk info for each run
for run in runs:
    for axis in axes:
        run[axis] = []
    for iteration in np.arange(start_iteration, end_iteration + increment, increment):
        c = CombinedFile(
            path=run['path'],
            iteration=iteration,
            number_of_processes=run['num_processes'],
            to_fname=f"merged_{iteration}_{randint(1, int(1e5))}.dat"
        )
        combined_file = c.combine_to_memory()
        # replace the headers
        combined_file.columns = file_headers
        time = c.sim_time
        # create the particle map
        particle_map = ParticleMap(particles=combined_file, mass_planet=mass_planet, equatorial_radius=equatorial_radius)
        particles = particle_map.loop()
        disk_particles = particles[particles['label'] == 'DISK']
        run['times'].append(time)
        run['disk_mass'].append(disk_particles['mass'].sum())
        run['disk_angular_momentum'].append(disk_particles['angular momentum'].sum())
        run['disk_entropy'].append(disk_particles['entropy'].mean())
        run['disk_impactor_mass_fraction'].append(disk_particles[disk_particles['tag'] > 1]['mass'].sum() / disk_particles['mass'].sum())
        run['disk_temperature'].append(disk_particles['temperature'].mean())
        run['disk_vmf'].append(None)
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex='all')
axs = axs.flatten()
for ax, (axis, ylabel) in zip(axs, zip(axes[1:], ylabels)):
    for run in runs:
        ax.plot(run['times'], run[axis], linewidth=2.0, label=run['name'])
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Time (hrs.)", fontsize=16)
    ax.grid()
    # use 16 point font on the axes
    ax.tick_params(axis='both', which='major', labelsize=16)
letters = string.ascii_lowercase
# annotate the plots with letters in the upper left corner
for ax, letter in zip(axs, letters):
    ax.text(0.05, 0.95, f"{letter}", transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
axs[0].legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"disk_profile.png", dpi=200)
