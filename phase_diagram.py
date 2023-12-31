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
    # {
    #     "name": "Canonical",
    #     "path": "/Users/scotthull/Desktop/canonical_df2.csv",
    #     "num_processes": 400,
    # },
    # {
    #     "name": "Half-Earths",
    #     "path": "/Users/scotthull/Desktop/half_earths_df2.csv",
    #     "num_processes": 400,
    # },
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

fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex='all', sharey='all')
axs = ax.flatten()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for ax in axs[-2:]:
    ax.set_xlabel("Entropy (J/kg/K)", fontsize=18)
for ax in [axs[0], axs[2]]:
    ax.set_ylabel("Temperature (K)", fontsize=18)

for ax in axs:
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
        label="100% Vapor"
    )
    ax.fill_between(
        x=phase_curve['entropy_sol_liq'],
        y1=phase_curve['temperature'],
        y2=critical_point,
        color=colors[-2],
        alpha=0.2,
        label="100% Liquid"
    )
    ax.fill_between(
        x=sorted(list(phase_curve['entropy_sol_liq']) + list(phase_curve['entropy_vap'])),
        y1=critical_point,
        y2=1e10,
        color=colors[-3],
        alpha=0.2,
        label="Supercritical"
    )
    ax.fill_between(
        x=phase_curve['entropy_sol_liq'],
        y1=phase_curve['temperature'],
        color=colors[-4],
        edgecolor="none",
        alpha=0.2,
        label="Liquid-Vapor Mixture"
    )
    ax.fill_between(
        x=phase_curve['entropy_vap'],
        y1=phase_curve['temperature'],
        color=colors[-4],
        edgecolor="none",
        alpha=0.2,
    )

for index, run in enumerate(runs):
    axs[index].set_title(f"{run['name']}", fontsize=20)
    # create the combined file
    c = CombinedFile(
        path=run['path'],
        iteration=run['final_iteration'],
        # iteration=iteration,
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
    final_disk_particles = particles[particles['label'] == 'DISK']
    final_disk_particles = final_disk_particles[final_disk_particles['tag'] % 2 == 0]

    c = CombinedFile(
        path=run['path'],
        iteration=run['final_iteration'],
        # iteration=iteration,
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{iteration}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    disk_particles = combined_file[combined_file['id'].isin(final_disk_particles['id']).tolist()]

    # disk_particles = pd.read_csv(run['path'])
    # axs[index].scatter(
    #     disk_particles['total entropy'], disk_particles['temperature'], s=5, alpha=1, label=run['name']
    # )
    axs[index].scatter(
        disk_particles['entropy'], disk_particles['temperature'], s=20, alpha=1, color='red'
    )
    # add text in lower right hand corner saying how many particles there are
    axs[index].text(0.80, 0.10, f"{len(disk_particles)} particles", transform=axs[index].transAxes,
                    ha='center', va="center", fontsize=18, weight='bold')
    axs[index].axvline(disk_particles['entropy'].mean(), color='black', linestyle='--')
    axs[index].axhline(disk_particles['temperature'].mean(), color='black', linestyle='--')

for ax in axs:
    ax.set_xlim(1800, 12000)
    ax.set_ylim(0, 12500)
    ax.grid(alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=14)
axs[0].legend(fontsize=18)
plt.tight_layout()
plt.savefig("phase_diagram.png", dpi=200)
