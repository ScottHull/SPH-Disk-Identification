#!/usr/bin/env python3
import pandas as pd
from random import randint
import matplotlib.pyplot as plt

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
        'final_iteration': 360,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "B",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b073_1v_esc/500_mars_b073_1v_esc",
        "num_processes": 600,
        'final_iteration': 360,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "C",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/500_mars_b050_1v_esc/500_mars_b050_1v_esc",
        "num_processes": 600,
        'final_iteration': 360,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "F",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_n_sph/mars_b073_1v_esc/mars_b073_1v_esc",
        "num_processes": 600,
        'final_iteration': 360,
        'phase_curve': "src/phase_curves/duniteN_vapour_curve.txt",
    },
]

iterations = [20, 100, 200, 300, 360]

# define the dataframe headers
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
# define the planet parameters
mass_mars = 6.39e23
equatorial_radius = 3390e3

# make a figure with len(runs) columns and len(iterations) rows, and scale the figure size accordingly
fig, ax = plt.subplots(len(runs), len(iterations), figsize=(len(iterations) * 5, len(runs) * 5), sharex='all',
                       sharey='all')

for run_index, run in enumerate(runs):
    # generate the end state data
    c = CombinedFile(
        path=run['path'],
        iteration=run['final_iteration'],
        num_processes=run['num_processes'],
        filetype='output'
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    time = c.sim_time
    # create the particle map
    particle_map = ParticleMap(particles=combined_file, mass_planet=mass_mars, equatorial_radius=equatorial_radius)
    endstate_particles = particle_map.loop()[['id', 'label']]  # drop all columns except "id" and "label"

    # loop through iterations
    for time_index, i in enumerate(iterations):
        # generate the data
        c = CombinedFile(
            path=run['path'],
            iteration=i,
            num_processes=run['num_processes'],
            filetype='output'
        )
        combined_file = c.combine_to_memory()
        # replace the headers
        combined_file.columns = file_headers
        time = c.sim_time

        for label_index, l in enumerate(['DISK', 'PLANET', 'ESCAPE']):
            label = None
            if run_index == time_index == 0:
                label = l.title()
            # plot the particles
            ax[run_index, time_index].scatter(
                combined_file[combined_file['label'] == 'DISK']['x'],
                combined_file[combined_file['label'] == 'DISK']['y'],
                marker='.',
                s=6,
                alpha=1,
                label=l
            )
