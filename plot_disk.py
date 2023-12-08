#!/usr/bin/env python3
import numpy as np
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

runs = [
    {
        "name": "A",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b073_2v_esc/500_mars_b073_2v_esc",
        "num_processes": 600,
        'final_iteration': 380,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "B",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b073_1v_esc/500_mars_b073_1v_esc",
        "num_processes": 600,
        'final_iteration': 380,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "C",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b050_1v_esc/500_mars_b050_1v_esc",
        "num_processes": 600,
        'final_iteration': 380,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "F",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_n_sph/500_mars_b073_1v_esc",
        "num_processes": 600,
        'final_iteration': 380,
        'phase_curve': "src/phase_curves/duniteN_vapour_curve.txt",
    },
]

iterations = [20, 100, 200, 300, 380]

# define the dataframe headers
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
# define the planet parameters
mass_mars = 6.39e23
equatorial_radius = 3390e3
square_scale = 4e7 / 10 ** 7

# make a figure with len(runs) columns and len(iterations) rows, and scale the figure size accordingly
fig, ax = plt.subplots(len(iterations), len(runs), figsize=(20, 24.5), sharex='all',
                       sharey='all')

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

        if time_index == 0:
            ax[time_index, run_index].set_title(f"{run['name']}", fontsize=20)
        if run_index == 0:
            ax[time_index, run_index].text(square_scale - (0.75 * square_scale), -square_scale + (0.3 * square_scale),
                                           f"{time} hrs.", fontsize=20)

        for label_index, l in enumerate(['PLANET', 'ESCAPE', 'DISK']):
            endstate = endstate_particles[endstate_particles['label'] == l]['id'].values
            # get combined file particles that are in the end state
            relevant_particles = combined_file[combined_file['id'].isin(endstate)]
            label = None
            if run_index == time_index == 0:
                label = l.title()
            # plot the particles
            ax[time_index, run_index].scatter(
                relevant_particles['x'] / (10 ** 7),
                relevant_particles['y'] / (10 ** 7),
                marker='.',
                s=6,
                alpha=1,
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
    a.text(0.05, 0.05, f"{letters[index]}", transform=a.transAxes, va="center", fontsize=16, weight='bold')
    a.set_xlim(-square_scale, square_scale)
    a.set_ylim(-square_scale, square_scale)
    a.axes.set_aspect('equal')
    # increase axis font size
    a.tick_params(axis='both', which='major', labelsize=16)

ax[0, 0].annotate(r"x ($10^4$ km)", xy=(0.0, -5.5), ha="center", fontsize=16, weight='bold')
ax[0, 0].annotate(r"y ($10^4$ km)", xy=(-5.5, 0.0), va="center", rotation=90, fontsize=16, weight='bold')
ax[0, 0].text(0.50, 0.05, r"x ($10^4$ km)", transform=ax[0, 0].transAxes, va="center", fontsize=14, weight='bold')
ax[0, 0].text(0.05, 0.5, r"y ($10^4$ km)", transform=ax[0, 0].transAxes, ha="center", rotation=90, fontsize=14, weight='bold')
# plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
axs = ax.flatten()
for ax in axs[-len(runs):-2]:
    nbins_x = len(ax.get_xticklabels())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins_x, prune='upper'))
for ax in [axs[i] for i in np.arange(len(runs) * 2, len(iterations) * len(runs), len(runs))]:
    nbins_y = len(ax.get_yticklabels())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins_y, prune='upper'))
# plt.tight_layout()
plt.savefig("source_scenes.png", format='png', dpi=200)
