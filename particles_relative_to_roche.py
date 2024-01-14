#!/usr/bin/env python3
import pandas as pd
import numpy as np
import string
from random import randint
import matplotlib.pyplot as plt

from src.combine import CombinedFile
from src.identify import ParticleMap
from src.report import GiantImpactReport

# use seaborn colorblind palette
plt.style.use('seaborn-colorblind')
# get the color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

runs = [
    {
        "name": "G",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 60,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "H",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 60,
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

# define the dataframe headers
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]

mass_mars = 6.39e23
equatorial_radius = 3390e3
roche_radius = 2.5 * equatorial_radius

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

ax.axvline(roche_radius, color='k', linestyle='--', label='Roche Limit')

for run in runs:
    # create the combined file
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
    particle_map = ParticleMap(particles=combined_file, mass_planet=mass_mars, equatorial_radius=equatorial_radius)
    particles = particle_map.loop()
    disk_particles = particles[particles['label'] == 'DISK']

    # create a CDF curve for the disk particles's radial position
    # get the radial position of the disk particles
    # sort the particles by their radial position
    disk_particles = disk_particles.sort_values(by='position')
    # create the CDF
    cdf = np.linspace(0, 1, len(disk_particles))
    # plot the CDF
    ax.plot(disk_particles['position'], cdf, linewidth=3.0, label=run['name'])

ax.set_xlabel("Radius (km)")
ax.set_ylabel("CDF")
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig("mars_cdf_roche_limit.pdf")

