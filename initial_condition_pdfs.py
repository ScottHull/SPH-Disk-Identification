#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import string
from random import randint
import matplotlib.pyplot as plt

from src.combine import CombinedFile
from src.identify import ParticleMap
from src.report import GiantImpactReport

# increase font size
plt.rcParams.update({'font.size': 18})
# use colorblind friendly colors
plt.style.use('seaborn-colorblind')


runs = [
    {
        "name": "G",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 35,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "H",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 35,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "K",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_diff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 60,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "L",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_diff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 60,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
]

# define the planet parameters
mass_mars = 6.39e23
equatorial_radius = 3390e3

file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]

# make a 4x4 plot
fig, axs = plt.subplots(4, 4, figsize=(16, 16))

# each run corresponds to a column
# plot a PDF of the velocity, entropy, temperature, and vmf for each run
for run_index, run in enumerate(runs):
    c = CombinedFile(
        path=run['path'],
        iteration=run['final_iteration'],
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{run['final_iteration']}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    # create the particle map
    particle_map = ParticleMap(particles=combined_file, mass_planet=mass_mars, equatorial_radius=equatorial_radius)
    endstate_particles = particle_map.loop()
    endstate_disk_particles = endstate_particles[endstate_particles['label'] == 'DISK']

    c = CombinedFile(
        path=run['path'],
        iteration=run['final_iteration'],
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{run['final_iteration']}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    time = c.sim_time
    # create the particle map
    disk_particles = combined_file[combined_file['id'].isin(endstate_disk_particles['id'].tolist())]
    disk_particles = disk_particles[disk_particles['tag'] % 2 == 0]
    disk_particles['velocity'] = np.sqrt(disk_particles['vx'] ** 2 + disk_particles['vy'] ** 2 + disk_particles['vz'] ** 2)

    axs[0, run_index].set_title(f"Run {run['name']}")

    axs[0, run_index].hist(disk_particles['velocity'] / 1000, bins=10, density=True)
    # twin the x axis and plot the CDF on the right axis
    ax2 = axs[0, run_index].twinx()
    sorted_data = disk_particles['velocity'].sort_values() / 1000
    cdf = sorted_data.rank(method='average', pct=True)
    ax2.plot(sorted_data, cdf, color='black', linewidth=3.0)
    if run_index == len(runs) - 1:
        ax2.set_ylabel("CDF")
    elif run_index == 0:
        axs[0, run_index].set_ylabel("Probability Density")
    axs[0, run_index].set_xlabel("Velocity (km/s)")

    axs[1, run_index].hist(disk_particles['entropy'] / 1000, bins=10, density=True)
    # twin the x axis and plot the CDF on the right axis
    ax2 = axs[1, run_index].twinx()
    sorted_data = disk_particles['entropy'].sort_values() / 1000
    cdf = sorted_data.rank(method='average', pct=True)
    ax2.plot(sorted_data, cdf, color='black', linewidth=3.0)
    if run_index == len(runs) - 1:
        ax2.set_ylabel("CDF")
    elif run_index == 0:
        axs[1, run_index].set_ylabel("Probability Density")
    axs[1, run_index].set_xlabel("Entropy (kJ/K/kg)")

    axs[2, run_index].hist(disk_particles['temperature'] / 1000, bins=10, density=True)
    # twin the x axis and plot the CDF on the right axis
    ax2 = axs[2, run_index].twinx()
    sorted_data = disk_particles['temperature'].sort_values() / 1000
    cdf = sorted_data.rank(method='average', pct=True)
    ax2.plot(sorted_data, cdf, color='black', linewidth=3.0)
    if run_index == len(runs) - 1:
        ax2.set_ylabel("CDF")
    elif run_index == 0:
        axs[2, run_index].set_ylabel("Probability Density")
    axs[2, run_index].set_xlabel("Temperature (K)")

    axs[3, run_index].hist(disk_particles['vmf_wo_circ'] * 100, bins=10, density=True)
    # twin the x axis and plot the CDF on the right axis
    ax2 = axs[3, run_index].twinx()
    sorted_data = disk_particles['vmf_wo_circ'].sort_values() * 100
    cdf = sorted_data.rank(method='average', pct=True)
    ax2.plot(sorted_data, cdf, color='black', linewidth=3.0)
    if run_index == len(runs) - 1:
        ax2.set_ylabel("CDF")
    elif run_index == 0:
        axs[3, run_index].set_ylabel("Probability Density")
    axs[3, run_index].set_xlabel("VMF (%)")

axs = axs.flatten()
for ax in axs:
    ax.grid()

plt.tight_layout()
plt.savefig("disk_initial_condition_pdfs.png", format='png', dpi=200)
