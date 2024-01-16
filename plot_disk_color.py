#!/usr/bin/env python3
import numpy as np
import pandas
import pandas as pd
import string
from math import log10
from random import randint
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.combine import CombinedFile
from src.identify import ParticleMap
from src.report import GiantImpactReport

target_param = 'entropy'
verbose_name = r"$\Delta S$ (1000 J/kg/K)"
min_normalize = 0
max_normalize = 2
target_param_norm = 1000
square_scale = 1

# normalizer = LogNorm(min_normalize, max_normalize)
normalizer = Normalize(min_normalize, max_normalize)
cmap = cm.get_cmap('cool')

runs = [
    {
        "name": "G",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_undiff",
        "num_processes": 400,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "H",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff",
        "num_processes": 400,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "K",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1vesc_b073_stewart_diff",
        "num_processes": 400,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "L",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_diff",
        "num_processes": 400,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
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
]

# iterations = [50, 100, 200, 300, 360]
# iterations = [50, 200, 500, 1000, 1800]
iterations = [15, 20, 25, 1000, 1800]

# define the dataframe headers
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]

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
fig, axs = plt.subplots(len(iterations), len(runs), figsize=(20, 24.5), sharex='all',
                       sharey='all', gridspec_kw=dict(hspace=0, wspace=0))

for run_index, run in enumerate(runs):
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
            axs[time_index, run_index].set_title(f"{run['name']}", fontsize=22)
        if run_index == 0:
            axs[time_index, run_index].text(0.68, 0.22, f"{time} hrs.", transform=axs[time_index, run_index].transAxes, ha='center',
                                           va="center", fontsize=20, weight='bold')


        # plot the particles
        axs[time_index, run_index].scatter(
            (combined_file['x'] - com_x) / (10 ** 7),  # to km and units of the square scale
            (combined_file['y'] - com_y) / (10 ** 7),
            c=cmap(normalizer((combined_file[target_param] - 3165) / target_param_norm)),
            marker='.',
            s=5,
            alpha=1,
        )

sm = cm.ScalarMappable(norm=normalizer, cmap=cmap)
sm.set_array([])
cbaxes = inset_axes(axs[0, len(runs) - 1], width="70%", height="10%", loc=1, borderpad=2)
cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_title(verbose_name, fontsize=16)

letters = list(string.ascii_lowercase)
for index, a in enumerate(axs.flatten()):
    a.text(0.05, 0.08, f"{letters[index]}", transform=a.transAxes, va="center", fontsize=22, weight='bold')
    if a <= 11:
        a.set_xlim(-square_scale, square_scale)
        a.set_ylim(-square_scale, square_scale)
    else:
        a.set_xlim(-square_scale * 4, square_scale * 4)
        a.set_ylim(-square_scale * 4, square_scale * 4)
    a.axes.set_aspect('equal')
    # increase axis font size
    a.tick_params(axis='both', which='major', labelsize=20)

axs[0, 0].text(0.50, 0.05, r"x ($10^4$ km)", transform=axs[0, 0].transAxes, ha="center", fontsize=20, weight='bold')
axs[0, 0].text(0.05, 0.5, r"y ($10^4$ km)", transform=axs[0, 0].transAxes, va="center", rotation=90, fontsize=20,
              weight='bold')
# plt.tight_layout()
# fig.subplots_adjust(wspace=0, hspace=0)
# plt.subplots_adjust(top=1, bottom=2, right=2, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
axs = axs.flatten()
for ax in axs[-len(runs):-2]:
    nbins_x = len(ax.get_xticklabels())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins_x, prune='upper'))
for ax in [axs[i] for i in np.arange(len(runs) * 2, len(iterations) * len(runs), len(runs))]:
    nbins_y = len(ax.get_yticklabels())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins_y, prune='upper'))
plt.tight_layout()
plt.savefig("source_scenes_colored.png", format='png', dpi=200)
