#!/usr/bin/env python3
import numpy as np
import pandas
import pandas as pd
import string
from random import randint
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.combine import CombinedFile
from src.identify import ParticleMap
from src.report import GiantImpactReport

# use the dark background
plt.style.use('dark_background')
# get the color cycle as a list
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

planet_color = colors[0]
escape_color = colors[1]
disk_color = colors[3]

runs = [
    # {
    #     "name": "A",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b073_2v_esc/500_mars_b073_2v_esc",
    #     "num_processes": 600,
    #     'final_iteration': 360,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
    # {
    #     "name": "B",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b073_1v_esc/500_mars_b073_1v_esc",
    #     "num_processes": 600,
    #     'final_iteration': 360,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
    # {
    #     "name": "C",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b050_1v_esc/500_mars_b050_1v_esc",
    #     "num_processes": 600,
    #     'final_iteration': 360,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
    # {
    #     "name": "F",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_n_sph/500_mars_b073_1v_esc",
    #     "num_processes": 600,
    #     'final_iteration': 360,
    #     'phase_curve': "src/phase_curves/duniteN_vapour_curve.txt",
    # },
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

# iterations = [50, 100, 200, 300, 360]
iterations = [50, 200, 500, 1000, 1800]

# define the dataframe headers
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
# define the planet parameters
mass_mars = 6.39e23
equatorial_radius = 3390e3
square_scale = 4

def center_of_mass(particles: pandas.DataFrame):
    """
    Calculate the center of mass of the particles
    """
    # calculate the center of mass
    x = particles['x'].values
    y = particles['y'].values
    z = particles['z'].values
    m = particles['mass'].values
    x_cm = np.sum(x * m) / np.sum(m)
    y_cm = np.sum(y * m) / np.sum(m)
    z_cm = np.sum(z * m) / np.sum(m)
    return x_cm, y_cm, z_cm

# make a figure with len(runs) columns and len(iterations) rows, and scale the figure size accordingly
fig, ax = plt.subplots(len(iterations), len(runs), figsize=(20, 24.5), sharex='all',
                       sharey='all', gridspec_kw=dict(hspace=0, wspace=0))

for run_index, run in enumerate(runs):
    # generate the end state data
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
    endstate_particles = particle_map.loop()[['id', 'label']]  # drop all columns except "id" and "label"

    # loop through iterations
    for time_index, i in enumerate(iterations):
        # generate the data
        c = CombinedFile(
            path=run['path'],
            iteration=i,
            number_of_processes=run['num_processes'],
            to_fname=f"merged_{i}_{randint(1, int(1e5))}.dat"
        )
        combined_file = c.combine_to_memory()
        # replace the headers
        combined_file.columns = file_headers
        time = c.sim_time

        # center on core of target
        com_x, com_y, com_z = center_of_mass(combined_file[combined_file['tag'] == 1])

        if time_index == 0:
            ax[time_index, run_index].set_title(f"{run['name']}", fontsize=22)
        if run_index == 0:
            ax[time_index, run_index].text(0.68, 0.22, f"{time} hrs.", transform=ax[time_index, run_index].transAxes, ha='center',
                                           va="center", fontsize=20, weight='bold')

        for label_index, (l, color) in enumerate(zip(['PLANET', 'ESCAPE', 'DISK'], [planet_color, escape_color, disk_color])):
            s = 6
            if l == "DISK":
                s = 200
            endstate = endstate_particles[endstate_particles['label'] == l]['id'].values
            # get combined file particles that are in the end state
            relevant_particles = combined_file[combined_file['id'].isin(endstate)]
            label = None
            if run_index == time_index == 0:
                label = l.title()
            # plot the particles
            ax[time_index, run_index].scatter(
                (relevant_particles['x'] - com_x) / (10 ** 7),  # to km and units of the square scale
                (relevant_particles['y'] - com_y) / (10 ** 7),
                marker='.',
                s=s,
                alpha=1,
                color=color,
                label=l
            )

legend = ax[0, 0].legend(loc='upper right', fontsize=16)
for handle in legend.legendHandles:
    try:
        handle.set_sizes([200.0])
    except:
        pass

letters = list(string.ascii_lowercase)
for index, a in enumerate(ax.flatten()):
    # x1, x2, y1, y2 = a.axis()
    # x_loc = x1 + (0.02 * (x2 - x1))
    # y_loc = y2 - (0.08 * (y2 - y1))
    # a.text(x_loc, y_loc, letters[index], fontweight="bold", fontsize=20)
    a.text(0.05, 0.08, f"{letters[index]}", transform=a.transAxes, va="center", fontsize=22, weight='bold')
    a.set_xlim(-square_scale, square_scale)
    a.set_ylim(-square_scale, square_scale)
    a.axes.set_aspect('equal')
    # increase axis font size
    a.tick_params(axis='both', which='major', labelsize=20)

ax[0, 0].text(0.50, 0.05, r"x ($10^4$ km)", transform=ax[0, 0].transAxes, ha="center", fontsize=20, weight='bold')
ax[0, 0].text(0.05, 0.5, r"y ($10^4$ km)", transform=ax[0, 0].transAxes, va="center", rotation=90, fontsize=20,
              weight='bold')
# plt.tight_layout()
# fig.subplots_adjust(wspace=0, hspace=0)
# plt.subplots_adjust(top=1, bottom=2, right=2, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
axs = ax.flatten()
for ax in axs[-len(runs):-2]:
    nbins_x = len(ax.get_xticklabels())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins_x, prune='upper'))
for ax in [axs[i] for i in np.arange(len(runs) * 2, len(iterations) * len(runs), len(runs))]:
    nbins_y = len(ax.get_yticklabels())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins_y, prune='upper'))
plt.tight_layout()
plt.savefig("source_scenes.png", format='png', dpi=200)
