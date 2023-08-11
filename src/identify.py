import numpy as np
import pandas as pd

from src.elements import total_orbital_energy, angular_momentum, eccentricity, semi_major_axis


class ParticleMap:
    """
    Class for mapping particles to their respective location (planet, disk, or escaping particles.
    """

    def __init__(self, particles: pd.DataFrame, mass_grav_body: float, initial_equatorial_radius: float, center=True):
        self.center_of_mass = None
        self.particles = particles  # the dataframe of particles
        self.mass_grav_body = mass_grav_body  # the mass of the main planetary body
        self.initial_equatorial_radius = initial_equatorial_radius  # the initial equatorial radius of the planet
        self.center = center  # whether to center the particles around the center of mass of the planet

    def center_of_mass(self, particles: pd.DataFrame):
        """
        Calculate the center of mass.
        :return:
        """
        total_mass = particles['mass'].sum()
        x = (particles['mass'] * particles['x']).sum() / total_mass
        y = (particles['mass'] * particles['y']).sum() / total_mass
        z = (particles['mass'] * particles['z']).sum() / total_mass
        return x, y, z

    def planetary_center_of_mass(self, target_iron_id=1):
        """
        Calculate the center of mass (COM) of the planetary body.
        This is best done by taking the COM of the target iron, which doesn't get ejected typically.
        Override this function for really, really exotic impacts.
        :return:
        """
        target_iron = self.particles[self.particles['tag'] == target_iron_id]
        return self.center_of_mass(target_iron)

    def prepare_particles(self):
        """
        Run this to calculate some stuff before identifying particles that is typically not given by SPH outputs.
        :return:
        """
        if self.center:
            self.center_of_mass = self.planetary_center_of_mass()
            self.particles['x'] -= self.center_of_mass[0]
            self.particles['y'] -= self.center_of_mass[1]
            self.particles['z'] -= self.center_of_mass[2]

        self.particles['velocity'] = np.linalg.norm(self.particles[['vx', 'vy', 'vz']].values, axis=1)
        self.particles['position'] = np.linalg.norm(self.particles[['x', 'y', 'z']].values, axis=1)
        self.particles['orbital_energy'] = total_orbital_energy(self.particles, self.mass_grav_body)
        self.particles['angular_momentum'] = angular_momentum(self.particles)
        self.particles['eccentricity'] = eccentricity(self.particles)
        self.particles['circular_semi_major_axis'] = -self.particles['orbital_energy'] / (self.mass_grav_body ** 2)
        self.particles['label'] = None

    def is_planet(self, particle):
        """
        Determine if a particle is part of the planet.
        Determines if the position of the particle is currently within the initial equatorial radius of the planet.
        If so, it is by definition part of the planet.
        :param particle:
        :return:
        """
        if particle['position'] <= self.initial_equatorial_radius:
            particle['label'] = 'PLANET'

    def will_be_planet(self, particle):
        """
        Determine if a particle will become part of the planet.
        Determines if the semi-major axis of the circular orbit of the particle is within the initial equatorial radius
        of the planet.  If so, the particle will eventually accrete to the planet and become part of it.
        :return:
        """
        if particle['circular_semi_major_axis'] <= self.initial_equatorial_radius:
            particle['label'] = 'PLANET'


    def is_disk(self, particle):
        """
        If the periapsis of the particle is greater than the initial equatorial radius of the planet and it's not
        on a hyperbolic orbit, then it is part of the disk.
        :param particle:
        :return:
        """
        if particle['periapsis'] > self.initial_equatorial_radius and particle['eccentricity'] < 1:
            particle['label'] = 'DISK'

    def is_escaping(self, particle):
        """
        If the eccentricity of the particle is greater than 1 (i.e., hyperbolic orbit), then it is escaping.
        :param particle:
        :return:
        """
        if particle['eccentricity'] > 1:
            particle['label'] = 'ESCAPE'

    def identify(self):
        """
        The main function for identifying particles.
        :return:
        """
        self.prepare_particles()  # prepare the particles for identification
        # map the particles to their respective location
        self.particles.apply(self.is_planet, axis=1)
        self.particles.apply(self.will_be_planet, axis=1)
        self.particles.apply(self.is_disk, axis=1)

