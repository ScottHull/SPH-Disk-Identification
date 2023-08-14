import numpy as np
import pandas as pd

from src.vectorized_elements import (total_orbital_energy, angular_momentum, eccentricity, semi_major_axis,
                          equivalent_circular_semi_major_axis, angular_momentum_vector, z_angular_momentum_vector)


class ParticleMap:
    """
    Class for mapping particles to their respective location (planet, disk, or escaping particles.
    """

    def __init__(self, particles: pd.DataFrame, mass_planet: float, equatorial_radius: float, center=True):
        self.center_of_mass = None
        self.particles = particles  # the dataframe of particles
        self.mass_planet = mass_planet  # the mass of the main planetary body
        self.equatorial_radius = equatorial_radius  # the equatorial radius of the planet
        self.poloidal_radius = equatorial_radius  # the initial poloidal radius of the planet
        self.oblateness = None  # the oblateness of the planet
        self.center = center  # whether to center the particles around the center of mass of the planet
        self.f = None  # the oblateness factor of the planet
        self.bulk_density = None  # the bulk density of the planet
        self.has_converged = False  # whether the iterative solution has converged
        self.iterations = 0  # the number of iterations to converge
        self.error = 1e99  # the error of the iterative solution

    def check_convergence(self, equatorial_radius, target_error=10 ** -8):
        """
        Check if the iterative solution has converged.
        :param target_error: the target error to converge to.
        :param equatorial_radius: the equatorial radius of the planet.
        :return:
        """
        self.error = np.abs(equatorial_radius - self.equatorial_radius) / self.equatorial_radius
        if self.error <= target_error:
            self.has_converged = True
            return True

    def calculate_center_of_mass(self, particles: pd.DataFrame):
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
        return self.calculate_center_of_mass(target_iron)

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
        self.particles['label'] = None

    def planet_bulk_density(self):
        """
        Calculate the bulk density of the planet.
        :return:
        """
        return self.mass_planet / (4 / 3 * np.pi * self.equatorial_radius ** 3)

    def is_planet(self, particle):
        """
        Determine if a particle is part of the planet.
        Determines if the position of the particle is currently within the equatorial radius of the planet.
        If so, it is by definition part of the planet.
        :param particle:
        :return:
        """
        if particle['position'] <= self.equatorial_radius:
            particle['label'] = 'PLANET'

    def will_be_planet(self, particle):
        """
        Determine if a particle will become part of the planet.
        Determines if the semi-major axis of the circular orbit of the particle is within the equatorial radius
        of the planet.  If so, the particle will eventually accrete to the planet and become part of it.
        :return:
        """
        if particle['circular_semi_major_axis'] <= self.equatorial_radius:
            particle['label'] = 'PLANET'

    def is_disk_or_escaping(self, particle):
        """
        If the periapsis of the particle is greater than the equatorial radius of the planet and it's not
        on a hyperbolic orbit, then it is part of the disk.
        :param particle:
        :return:
        """
        if particle['periapsis'] > self.equatorial_radius and particle['eccentricity'] <= 1:
            particle['label'] = 'DISK'
        elif particle['eccentricity'] > 1:
            particle['label'] = 'ESCAPE'

    def is_planet_disk_or_escaping(self, particle):
        """
        Determine if a particle is part of the planet, disk, or escaping.
        """
        self.is_planet(particle)
        self.is_disk_or_escaping(particle)

    def roche_radius(self):
        """
        Get the Roche radius of the planet.
        :return: 
        """
        return 2.9 * self.equatorial_radius

    def calculate_elements(self):
        """
        Calculate all necessary orbital elements in order.
        :return:
        """
        self.particles['angular momentum vector'] = angular_momentum_vector(self.particles)
        self.particles['z angular momentum vector'] = z_angular_momentum_vector(self.particles)
        self.particles['angular momentum'] = angular_momentum(self.particles)
        self.particles['orbital energy'] = total_orbital_energy(self.particles, self.mass_planet)
        self.particles['eccentricity'] = eccentricity(self.particles, self.mass_planet)
        self.particles['semi major axis'] = semi_major_axis(self.particles, self.mass_planet)
        self.particles['circular semi major axis'] = equivalent_circular_semi_major_axis(self.particles,
                                                                                         self.mass_planet)

    def calculate_planetary_oblateness(self, K=0.335):
        """
        Calculate the oblateness of the planet.
        :param K:
        :return:
        """
        G = 6.67408e-11  # m^3 kg^-1 s^-2, gravitational constant
        z_angular_momentum_planet = self.particles['z_angular_momentum_vector'].sum()
        moment_of_inertia_planet = (2.0 / 5.0) * self.mass_planet * self.equatorial_radius ** 2
        angular_velocity_protoplanet = z_angular_momentum_planet / moment_of_inertia_planet
        keplerian_velocity_protoplanet = np.sqrt(G * self.mass_planet / self.equatorial_radius ** 3)
        numerator = (5.0 / 2.0) * ((angular_velocity_protoplanet / keplerian_velocity_protoplanet) ** 2)
        denominator = 1.0 + ((5.0 / 2.0) - ((15.0 * K) / 4.0)) ** 2
        return numerator / denominator

    def calculate_planetary_radii(self, mass_planet):
        """
        Calculate the equatorial and polar radii of the planet.
        :param mass_planet:
        :return:
        """
        self.oblateness = self.calculate_planetary_oblateness()
        return ((3 * mass_planet) / (4 * np.pi * self.bulk_density * (1 - self.oblateness))) ** (1 / 3)

    def identify(self):
        """
        The main function for identifying particles.
        :return:
        """
        self.prepare_particles()  # prepare the particles for identification
        while not self.has_converged:
            print(f"Beginning convergence iteration {self.iterations}")
            # calculate orbital elements
            print("Calculating orbital elements...")
            self.calculate_elements()
            print("Calculating orbital elements complete.")
            # map the particles to their respective location
            print("Mapping particles to their respective location...")
            self.particles.apply(self.is_planet_disk_or_escaping, axis=1)
            print("Mapping particles to their respective location complete.")
            # calculate the new oblateness, planet mass, and equatorial radius
            mass_planet = self.particles[self.particles['label'] == 'PLANET']['mass'].sum()
            equatorial_radius = self.calculate_planetary_radii(mass_planet)
            self.error = np.abs(equatorial_radius - self.equatorial_radius) / self.equatorial_radius
            # check for solution convergence
            self.has_converged = self.check_convergence(equatorial_radius)
            # update the planet mass and equatorial radius
            self.mass_planet = mass_planet
            self.equatorial_radius = equatorial_radius
            self.poloidal_radius = self.poloidal_radius * (1 - self.oblateness)
            print(f"Convergence iteration {self.iterations} complete (error: {self.error}).")
            self.iterations += 1

    def loop(self, num_iterations=2):
        """
        Loop over the identification process until convergence.
        :param num_iterations:
        :return:
        """
        for i in range(num_iterations):
            self.bulk_density = self.planet_bulk_density()
            self.identify()
        return self.particles
