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
# increase font size
plt.rcParams.update({'font.size': 18})

runs = [
    {
        "name": "G",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 27,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "H",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 19,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "K",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_diff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 27,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "L",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_diff",
        "num_processes": 400,
        'final_iteration': 1800,
        'max_vel_profile_iteration': 20,
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
        # iteration=run['final_iteration'],
        iteration=run['max_vel_profile_iteration'],
        # iteration=iteration,
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{iteration}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    # particle_map = ParticleMap(particles=combined_file, mass_planet=mass_planet, equatorial_radius=equatorial_radius)
    # combined_file = particle_map.loop()
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
    # axs[index].axvline(disk_particles['total_entropy'].mean(), color='black', linestyle='--')
    axs[index].axhline(disk_particles['temperature'].mean(), color='black', linestyle='--')

for ax in axs:
    ax.set_xlim(1800, 12000)
    ax.set_ylim(0, 12500)
    ax.grid(alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=14)
axs[0].legend(fontsize=18)
plt.tight_layout()
plt.savefig("phase_diagram.png", dpi=200)


# we're going to do a delta entropy plot for the initial condition
MELT_THRESHOLD = 623  # delta S, J/kg/K
silicate_S = 3165
metal_S = 1500
disk_delta_S = {}

for index, run in enumerate(runs):
    disk_delta_S[f"{run['name']}"] = {}
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
        iteration=run['max_vel_profile_iteration'],
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{iteration}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    disk_particles_ic = combined_file[combined_file['id'].isin(final_disk_particles['id']).tolist()]
    disk_delta_S[f"{run['name']}"].update({'initial conditions': np.array(disk_particles_ic['entropy'] - silicate_S)})
    disk_delta_S[f"{run['name']}"].update({'final disk wo circ': np.array(final_disk_particles['entropy'] - silicate_S)})
    disk_delta_S[f"{run['name']}"].update({'final disk w circ': np.array(final_disk_particles['total entropy'] - silicate_S)})

# make a 3 column 1 row figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex='all', sharey='all')
axs = axs.flatten()
frac_above_delta_S = {'initial conditions': {}, 'final disk wo circ': {}, 'final disk w circ': {}}

for run_index, run in enumerate(runs):
    df = pd.DataFrame()
    for index, key in enumerate(disk_delta_S[f"{run['name']}"].keys()):
        # sort the delta_s list
        delta_s = np.sort(np.array(disk_delta_S[f"{run['name']}"][key]))
        frac_delta_s_above_threshold = len(delta_s[delta_s >= MELT_THRESHOLD]) / len(delta_s) * 100
        frac_above_delta_S[key][run['name']] = frac_delta_s_above_threshold
        # make a CDF
        cdf = np.arange(1, len(delta_s) + 1) / len(delta_s)
        # plot a CDF of the delta S
        axs[index].plot(
            delta_s, cdf, linewidth=3.0, color=colors[run_index], label=f"Run {run['name']}"
        )

for index, key in enumerate(frac_above_delta_S.keys()):
    s = r"$\rm \Delta S > $" + f"{MELT_THRESHOLD} J/kg/K"
    for run in frac_above_delta_S[key].keys():
        s += f"\nRun {run}: {frac_above_delta_S[key][run]:.2f} %"
    # annotate frac above threshold in lower right corner of each plot
    axs[index].text(0.50, 0.30, s, transform=axs[index].transAxes, ha='left', va="center", fontsize=18)

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs):
    ax.set_xlabel(r"$\rm \Delta S$ (J/kg/K)", fontsize=18)
    ax.axvline(MELT_THRESHOLD, color='black', linestyle='--', linewidth=3.0)
    ax.grid()
    ax.text(
        0.95, 0.05, letters[index], ha='center', va="center", transform=ax.transAxes, fontsize=20, weight='bold'
    )

for label, ax in zip(['Initial Jet Conditions', 'End-State Disk (w/o circ.)', 'End-State Disk (w/ circ.)'], axs):
    ax.set_title(label, fontsize=18)

axs[0].set_ylabel("CDF")
legend = axs[0].legend(fontsize=18, loc='upper right')
# increase the linewidth of the lines in the legend
for line in legend.get_lines():
    line.set_linewidth(5.0)
plt.tight_layout()
plt.savefig("delta_s_cdf.png", format='png', dpi=200)
