#!/usr/bin/env python3
import numpy as np
import pandas as pd
import string
from random import randint
import matplotlib.pyplot as plt

from src.combine import CombinedFile
from src.identify import ParticleMap
from src.report import GiantImpactReport


runs = [
    {
        "name": "H",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
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

to_track = ['iterations', 'times', 'velocities', 'temperatures', 'vmfs']

# fig, axs = plt.subplots(len(runs), 3, figsize=(20, len(runs) * 3), sharex='all', sharey='all')
fig, axs = plt.subplots(2, 3, figsize=(20, len(runs) * 3), sharex='all', sharey='all')

for index, run in enumerate(runs):
    for t in to_track:
        run[t] = []
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
    endstate_particles = particle_map.loop()
    endstate_disk_particles = endstate_particles[endstate_particles['label'] == 'DISK']

    for iteration in np.arange(0, run['max_vel_profile_iteration'] + 1, 1):
        c = CombinedFile(
            path=run['path'],
            iteration=iteration,
            number_of_processes=run['num_processes'],
            to_fname=f"merged_{run['final_iteration']}_{randint(1, int(1e5))}.dat"
        )
        combined_file = c.combine_to_memory()
        # replace the headers
        combined_file.columns = file_headers
        time = c.sim_time
        disk_particles = combined_file[combined_file['id'].isin(endstate_disk_particles['id'].tolist())]
        disk_particles['velocity'] = np.sqrt(disk_particles['vx'] ** 2 + disk_particles['vy'] ** 2 + disk_particles['vz'] ** 2)

        # generate the report
        phase_curve = pd.read_fwf(run['phase_curve'], skiprows=1,
                                  names=["temperature", "density_sol_liq", "density_vap", "pressure",
                                         "entropy_sol_liq", "entropy_vap"])
        r = GiantImpactReport()
        vmf_w_circ, vmf_wo_circ = r.calculate_vmf(particles=disk_particles[disk_particles['tag'] % 2 == 0],
                                                  phase_curve=phase_curve, entropy_col='entropy')

        run['iterations'].append(iteration)
        run['times'].append(time)
        run['velocities'].append(disk_particles['velocity'].mean())
        run['temperatures'].append(disk_particles['temperature'].mean())
        run['vmfs'].append(vmf_wo_circ)

    max_iteration = run['velocities'].index(max(run['velocities']))
    c = CombinedFile(
        path=run['path'],
        iteration=max_iteration,
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{run['final_iteration']}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    time = c.sim_time
    disk_particles = combined_file[combined_file['id'].isin(endstate_disk_particles['id'].tolist())]


    axs[index, 0].scatter(
        combined_file['x'] / 1000, combined_file['y'] / 1000, s=2, marker=".", color='black'
    )
    axs[index, 0].scatter(
        disk_particles['x'] / 1000, disk_particles['y'] / 1000, s=2, marker=".", color='red'
    )
    axs[index, 0].set_xlabel("x (km)", fontsize=18)
    axs[index, 0].set_ylabel("y (km)", fontsize=18)
    axs[index, 1].plot(
        run['times'], np.array(run['velocities']) / 1000, linewidth=2.0, color='black'
    )
    axs[index, 1].set_xlabel("Time (hrs.)", fontsize=18)
    axs[index, 1].set_ylabel("Avg. Ejecta Velocity (km/s)", fontsize=18)
    axs[index, 1].axvline(run['times'][max_iteration], color='black', linestyle='--')
    axs[index, 1].text(
        0.6, 0.9, r"$t_{\rm ic} = $" + f"{run['times'][max_iteration]} hrs.", transform=axs[index, 1].transAxes, size=20
    )
    axs[index, 2].plot(
        run['times'], run['temperatures'], linewidth=2.0, color='blue'
    )
    ax2 = axs[index, 2].twinx()
    axs[index, 2].plot(
        run['times'], run['vmfs'], linewidth=2.0, color='red'
    )
    axs[index, 2].axvline(run['times'][max_iteration], color='black', linestyle='--')
    axs[index, 2].set_xlabel("Time (hrs.)", fontsize=18)
    axs[index, 2].set_ylabel("Avg. Disk Temperature (K)", fontsize=18)
    ax2.set_ylabel("Disk VMF (%)", fontsize=18)
    axs[index, 0].text(
        0.7, 0.9, f"{run['name']}", transform=axs[index, 0].transAxes, size=20
    )

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs.flatten()):
    ax.grid(alpha=0.4)
    axs[index].text(
        0.05, 0.9, f"({letters[index]})", transform=axs[index].transAxes, size=20
    )

plt.savefig("hydrodynamic_initial_conditions.png", format='png', dpi=200)