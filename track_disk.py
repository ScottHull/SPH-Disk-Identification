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
    {
        "name": "I",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff_rho_c_5kgm3",
        "num_processes": 400,
        'final_iteration': 1800,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "J",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff_rho_c_5kgm3",
        "num_processes": 400,
        'final_iteration': 1800,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    # {
    #     "name": "K",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_diff",
    #     "num_processes": 400,
    #     'final_iteration': 1800,
    #     'max_vel_profile_iteration': 60,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
    # {
    #     "name": "L",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_diff",
    #     "num_processes": 400,
    #     'final_iteration': 1800,
    #     'max_vel_profile_iteration': 60,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
]

# define the iteration parameters
start_iteration = 0
end_iteration = 1800
increment = 20

# define the planet parameters
mass_planet = 6.39e23
equatorial_radius = 3390e3
mass_phobos = 1.0659e16
mass_deimos = 1.4762e15

# define some misc stuff
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
axes = ['times', 'disk_entropy_w_circ', 'disk_temperature', 'disk_vmf_w_circ', 'disk_mass', 'disk_angular_momentum',
        'disk_impactor_mass_fraction', 'disk_vmf_wo_circ', 'disk_entropy_wo_circ']
ylabels = ["Avg. Disk Entropy (J/kg/K)", "Avg. Disk Temperature (K)", "Disk VMF (%)", r"Disk Mass ($10^3$ $M_{\rm PD}$)",
             r"$L_{\rm disk}^*$", r"$f_{\rm disk}$ (%)"]
phase_curve = pd.read_fwf("src/phase_curves/forstSTS__vapour_curve.txt", skiprows=1,
                           names=["temperature", "density_sol_liq", "density_vap", "pressure",
                                  "entropy_sol_liq", "entropy_vap"])
dr = GiantImpactReport()

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
        if len(disk_particles) > 0:
            disk_mass = disk_particles['mass'].sum() / (1000 * (mass_phobos + mass_deimos))
            disk_ang_mom = disk_particles['angular momentum'].sum()
            disk_impactor_mass_fraction = disk_particles[disk_particles['tag'] > 1]
            disk_impactor_mass_fraction = disk_impactor_mass_fraction['mass'].sum() / disk_particles['mass'].sum() * 100
            disk_entropy = disk_particles['entropy'].mean()
            disk_total_entropy = disk_particles['total entropy'].mean()
            disk_temperature = disk_particles['temperature'].mean()
        else:
            disk_mass = 0.0
            disk_ang_mom = 0.0
            disk_impactor_mass_fraction = 0.0
            disk_entropy = 0
            disk_total_entropy = 0
            disk_temperature = 0
        # scale the disk angular momentum
        if disk_mass > 0:
            L_scaled = disk_ang_mom / ((disk_mass * (1000 * (mass_phobos + mass_deimos))) * np.sqrt((6.67 * 10 ** -11) * mass_planet * 2.5 * equatorial_radius))
        else:
            L_scaled = 0.0
        disk_vmf_w_circ, disk_vmf_wo_circ = dr.calculate_vmf(disk_particles, phase_curve)
        run['times'].append(time)
        run['disk_mass'].append(disk_mass)
        run['disk_angular_momentum'].append(L_scaled)
        run['disk_entropy_w_circ'].append(disk_total_entropy)
        run['disk_entropy_wo_circ'].append(disk_entropy)
        run['disk_impactor_mass_fraction'].append(disk_impactor_mass_fraction)
        run['disk_temperature'].append(disk_temperature)
        run['disk_vmf_w_circ'].append(disk_vmf_w_circ)
        run['disk_vmf_wo_circ'].append(disk_vmf_wo_circ)

fig, axs = plt.subplots(2, 3, figsize=(18, 9), sharex='all')
axs = axs.flatten()
axs[1].set_ylim(1500, 3000)
axs[2].set_ylim(0, 30)
# axs[2].set_yscale('log')
for ax_index, (ax, (axis, ylabel)) in enumerate(zip(axs, zip(axes[1:-2], ylabels))):
    for index, run in enumerate(runs):
        label = None
        if ax_index == 0:
            label = f"Run {run['name']}"
        ax.plot(run['times'], run[axis], linewidth=2.0, color=colors[index], label=label)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Time (hrs.)", fontsize=16)
    ax.grid()
    # use 16 point font on the axes
    ax.tick_params(axis='both', which='major', labelsize=16)
for index, run in enumerate(runs):
    axs[0].plot(run['times'], run['disk_entropy_wo_circ'], linewidth=2.0, color=colors[index], linestyle="--")
    axs[2].plot(run['times'], run['disk_vmf_wo_circ'], linewidth=2.0, color=colors[index], linestyle="--")
letters = string.ascii_lowercase
# annotate the plots with letters in the upper left corner
for ax, letter in zip(axs, letters):
    ax.text(0.90, 0.95, f"{letter}", transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
plt.tight_layout()
legend = fig.legend(loc=7, fontsize=16)
for handle in legend.legendHandles:
    handle.set_linewidth(5.0)
fig.subplots_adjust(right=0.90)
plt.savefig(f"disk_profile.png", dpi=200)
