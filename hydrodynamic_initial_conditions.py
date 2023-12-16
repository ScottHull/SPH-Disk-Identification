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

to_track = ['iterations', 'times', 'velocities', 'temperatures', 'vmfs']
fname = "hydrodynamic_initial_conditions.csv"
if os.path.exists(fname):
    os.remove(fname)
with open(fname, 'w') as f:
    # write the header
    f.write("name,iteration,time,velocity,temperature,vmf,impactor_disk_mass_fraction\n")
f.close()

# fig, axs = plt.subplots(len(runs), 3, figsize=(20, len(runs) * 3), sharex='all', sharey='all')
fig, axs = plt.subplots(4, 3, figsize=(20, 20 * 3 / 4))

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
    impactor_disk_mass_fraction = endstate_disk_particles[endstate_disk_particles['tag'] > 1]['mass'].sum() / endstate_disk_particles['mass'].sum() * 100

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
        run['final_disk_particle_ids'] = disk_particles['id'].tolist()

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
    run['max_iteration'] = max_iteration
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
    disk_particles['velocity'] = np.sqrt(disk_particles['vx'] ** 2 + disk_particles['vy'] ** 2 + disk_particles['vz'] ** 2)
    run['disk_particles'] = disk_particles

    axs[index, 0].scatter(
        combined_file['x'] / 10 ** 6, combined_file['y'] / 10 ** 6, s=2, marker=".", color='black'
    )
    axs[index, 0].scatter(
        disk_particles['x'] / 10 ** 6, disk_particles['y'] / 10 ** 6, s=2, marker=".", color='red'
    )
    axs[index, 0].set_xlim(-8, 8)
    axs[index, 0].set_ylim(-8, 8)
    # axs[index, 0].set_xlabel("x (km)", fontsize=18)
    axs[index, 0].set_ylabel(r"y ($10^3$ km)", fontsize=18)
    axs[index, 1].plot(
        run['times'], np.array(run['velocities']) / 1000, linewidth=2.0, color='black'
    )
    # axs[index, 1].set_xlabel("Time (hrs.)", fontsize=18)
    axs[index, 1].set_ylabel("Avg. Ejecta Velocity (km/s)", fontsize=18)
    axs[index, 1].axvline(run['times'][max_iteration], color='black', linestyle='--')
    # axs[index, 1].text(
    #     0.6, 0.9, r"$t_{\rm ic} = $" + f"{run['times'][max_iteration]} hrs.", transform=axs[index, 1].transAxes, size=20
    # )
    axs[index, 2].plot(
        run['times'], run['temperatures'], linewidth=2.0, color='blue'
    )
    ax2 = axs[index, 2].twinx()
    ax2.plot(
        run['times'], run['vmfs'], linewidth=2.0, color='red'
    )
    axs[index, 2].axvline(run['times'][max_iteration], color='black', linestyle='--')
    # axs[index, 2].set_xlabel("Time (hrs.)", fontsize=18)
    axs[index, 2].set_ylabel("Avg. Disk Temperature (K)", fontsize=18)
    ax2.set_ylabel("Disk VMF (%)", fontsize=18)
    axs[index, 0].text(
        0.75, 0.9, f"Run {run['name']}", transform=axs[index, 0].transAxes, size=20
    )

    with open(fname, 'a') as f:
        f.write(
            run['name'] + "," + str(max_iteration) + "," + str(run['times'][max_iteration])+ "," + str(run['velocities'][max_iteration]) + "," +
            str(run['temperatures'][max_iteration]) + "," + str(run['vmfs'][max_iteration]) + "," + str(impactor_disk_mass_fraction) + "\n"
        )
    f.close()

letters = list(string.ascii_lowercase)
for index, ax in enumerate(axs.flatten()):
    ax.grid(alpha=0.4)
    ax.text(
        0.05, 0.9, f"{letters[index]}", transform=ax.transAxes, fontweight='bold', size=20
    )
    # increase the axis font size
    ax.tick_params(axis='both', which='major', labelsize=18)

for ax, label in zip(axs.flatten()[-3:], [r"x ($10^3$ km)", "Time (hrs.)", "Time (hrs.)"]):
    ax.set_xlabel(label, fontsize=18)

plt.tight_layout()
plt.savefig("hydrodynamic_initial_conditions.png", format='png', dpi=200)

#
# fig, axs = plt.subplots(4, 2, figsize=(10, 20))
#
# for index, run in enumerate(runs):
#     disk_particles = run['disk_particles']
#     disk_bound_particles = disk_particles[disk_particles['tag'] % 2 == 0]
#     phase_curve = pd.read_fwf(run['phase_curve'], skiprows=1,
#                               names=["temperature", "density_sol_liq", "density_vap", "pressure",
#                                      "entropy_sol_liq", "entropy_vap"])
#     vmf_w_circ, vmf_wo_circ = GiantImpactReport().calculate_vmf(particles=disk_bound_particles,
#                                               phase_curve=phase_curve, entropy_col='entropy')
#     # sort disk particles by vmf
#     disk_bound_particles = disk_bound_particles.sort_values(by=['vmf_wo_circ'], ascending=False)
#
#     axs[index, 0].scatter(
#         disk_bound_particles['velocity'] / 1000, disk_bound_particles['vmf_wo_circ'] * 100, s=5, marker="."
#     )
#     axs[index, 0].set_title(f"Run {run['name']}", fontsize=20)
#     axs[index, 0].set_ylabel("VMF (%)", fontsize=18)
#     axs[index, 1].plot(
#         disk_bound_particles['vmf_wo_circ'] * 100,
#         disk_bound_particles['vmf_wo_circ'].rank(method='average', pct=True), linewidth=2.0, color='black'
#     )
#     axs[index, 1].set_ylabel("CDF", fontsize=18)
#
#     axs[index, 0].axvline(
#         disk_bound_particles['velocity'].mean() / 1000, color='black', linestyle='--', linewidth=2.0
#     )
#     axs[index, 0].axhline(
#         disk_bound_particles['vmf_wo_circ'].sum() / len(disk_bound_particles['vmf_wo_circ']) * 100,
#         color='black', linestyle='--', linewidth=2.0
#     )
#     axs[index, 1].axvline(
#         disk_bound_particles['vmf_wo_circ'].sum() / len(disk_bound_particles['vmf_wo_circ']) * 100,
#         color='black', linestyle='--', linewidth=2.0
#     )
#
# for ax in axs.flatten():
#     ax.grid(alpha=0.4)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#
# axs = axs.flatten()
# axs[-2].set_xlabel("Velocity (km/s)", fontsize=18)
# axs[-1].set_xlabel("VMF (%)", fontsize=18)
# plt.tight_layout()
# plt.savefig("hydrodynamic_initial_conditions_velocity_vs_vmf.png", format='png', dpi=200)
