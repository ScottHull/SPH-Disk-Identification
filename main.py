import pandas as pd
from random import randint

from src.combine import CombinedFile
from src.identify import ParticleMap

# define where the data is
path = '/home/theia/scotthull/Paper2_SPH/gi/500_mars/500_mars'
number_of_proceses = 600
iteration = 500

# define the planet parameters
mass_planet = 6.39e23
equatorial_radius = 3390e3

# create the combined file
to_fname = f"merged_{iteration}_{randint(1, int(1e5))}.dat"
combined_file = CombinedFile(
    path=path,
    iteration=iteration,
    number_of_processes=number_of_proceses,
    to_fname=to_fname
).combine_to_memory()

# create the particle map
particle_map = ParticleMap(particles=combined_file, mass_planet=mass_planet, equatorial_radius=equatorial_radius)

print("Particle map created!")
print(
    f"Planet mass: {particle_map.mass_planet}\n"
    f"Planet radius: {particle_map.equatorial_radius}\n"
    f"Planet density: {particle_map.bulk_density}\n"
    f"# escaping particles: {len(particle_map[particle_map['label'] == 'ESCAPE'])}\n"
    f"# disk particles: {len(particle_map[particle_map['label'] == 'DISK'])}\n"
    f"# planet particles: {len(particle_map[particle_map['label'] == 'PLANET'])}\n"
)
