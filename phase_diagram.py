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

runs = [
    # {
    #     "name": "mars_citron_1vesc_b073_stewart_undiff",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
    #     "num_processes": 400,
    # },
    {
        "name": "mars_citron_1.4vesc_b073_stewart_undiff",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff",
        "num_processes": 400,
    },
]

iteration = 1800

# define the planet parameters
mass_planet = 6.39e23
equatorial_radius = 3390e3

file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
phase_curve = pd.read_fwf("src/phase_curves/forstSTS__vapour_curve.txt", skiprows=1,
                           names=["temperature", "density_sol_liq", "density_vap", "pressure",
                                  "entropy_sol_liq", "entropy_vap"])
critical_point = max(phase_curve['temperature'])

fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex='all', sharey='all')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax.set_xlabel("Entropy (J/kg/K)")
ax.set_ylabel("Temperature (K)")

# shade the phase diagram
ax.plot(
    phase_curve['entropy_sol_liq'],
    phase_curve['temperature'],
    linewidth=2.0,
    color='black'
)
ax.plot(
    phase_curve['entropy_vap'],
    phase_curve['temperature'],
    linewidth=2.0,
    color='black'
)
ax.fill_between(
    x=phase_curve['entropy_vap'],
    y1=phase_curve['temperature'],
    y2=critical_point,
    color=colors[-1],
    alpha=0.2,
    # label="100% Vapor"
)
ax.fill_between(
    x=phase_curve['entropy_sol_liq'],
    y1=phase_curve['temperature'],
    y2=critical_point,
    color=colors[-2],
    alpha=0.2,
    # label="100% Liquid"
)
ax.fill_between(
    x=sorted(list(phase_curve['entropy_sol_liq']) + list(phase_curve['entropy_vap'])),
    y1=critical_point,
    y2=1e10,
    color=colors[-3],
    alpha=0.2,
    # label="Supercritical"
)
ax.fill_between(
    x=phase_curve['entropy_sol_liq'],
    y1=phase_curve['temperature'],
    color=colors[-4],
    edgecolor="none",
    alpha=0.2,
    # label="Mixed"
)
ax.fill_between(
    x=phase_curve['entropy_vap'],
    y1=phase_curve['temperature'],
    color=colors[-4],
    edgecolor="none",
    alpha=0.2,
)

for run in runs:
    # create the combined file
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
    ax.scatter(
        disk_particles['total entropy'], disk_particles['temperature'], s=1, alpha=1, label=run['name']
    )

ax.set_xlim(1800, 12000)
ax.set_ylim(0, 12500)
ax.legend()
plt.savefig("phase_diagram.png", dpi=200)
