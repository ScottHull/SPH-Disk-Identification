#!/usr/bin/env python3
import pandas as pd
from random import randint
import matplotlib.pyplot as plt

from src.combine import CombinedFile
from src.identify import ParticleMap

# define where the data is
path = '/home/theia/scotthull/Paper2_SPH/gi/500_mars/500_mars'
start_iteration = 100
end_iteration = 1800
increment = 50
number_of_proceses = 600
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]

# define the planet parameters
mass_planet = 6.39e23
equatorial_radius = 3390e3

for iteration in range(start_iteration, end_iteration + 1, increment):
    # create the combined file
    c = CombinedFile(
        path=path,
        iteration=iteration,
        number_of_processes=number_of_proceses,
        to_fname=f"merged_{iteration}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    time = c.sim_time
    # create the particle map
    particle_map = ParticleMap(particles=combined_file, mass_planet=mass_planet, equatorial_radius=equatorial_radius)
    particles = particle_map.loop()

    fig = plt.figure(figsize=(10, 10))
    # use the dark background
    plt.style.use('dark_background')
    ax = fig.add_subplot(111)
    ax.set_title(f"{time} hrs. (Iteration {iteration})")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    # scatter planet, disk, and escaping particles
    for i in ['PLANET', 'DISK', 'ESCAPE']:
        ax.scatter(
            particles[particles['label'] == i]['x'] / 10 ** 7,
            particles[particles['label'] == i]['y'] / 10 ** 7,
            marker='.',
            s=2,
            label=i
        )
    legend = ax.legend(loc='upper right')
    # increase legend marker size
    for i in legend.legendHandles:
        i.set_sizes([20])

    plt.savefig(f"test_{iteration}.png", dpi=300)
