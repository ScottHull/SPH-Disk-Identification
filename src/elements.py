import numpy as np

G = 6.674 * 10 ** -11  # m^3 kg^-1 s^-2, the gravitational constant


def angular_momentum_vector(mass: float, position: np.array, velocity: np.array):
    """
    Returns a 3D vector of the angular momentum of a particle.
    :param mass:
    :param position:
    :param velocity:
    :return:
    """
    return mass * np.cross(position, velocity)


def angular_momentum(mass: float, position: np.array, velocity: np.array):
    """
    Returns the magnitude of the angular momentum of a particle.
    :param mass:
    :param position:
    :param velocity:
    :return:
    """
    return np.linalg.norm(angular_momentum_vector(mass, position, velocity))


def node_vector(position: np.array, angular_momentum_vector: np.array):
    """
    Returns the node vector of a particle.
    :param position:
    :param angular_momentum_vector:
    :return:
    """
    return np.cross([0, 0, position[2]], angular_momentum_vector)


def total_orbital_energy(mass: float, velocity: np.array, mass_grav_body: float, position: np.array):
    """
    Returns the total orbital energy of a particle.
    :param mass:
    :param velocity:
    :param mass_grav_body:
    :param position:
    :return:
    """
    # kinetic energy, KE = 1/2 m v^2
    kinetic_energy = (1.0 / 2.0) * mass * (np.linalg.norm(velocity) ** 2)
    # vectorized gravitational potential energy, PE = (G M_1 M_2) / r
    potential_energy = - (G * mass_grav_body * mass) / np.linalg.norm(position)
    return kinetic_energy + potential_energy


def eccentricity(mass: float, velocity: np.array, mass_grav_body: float, position: np.array,
                 orbital_energy: float, angular_momentum: float):
    """
    Returns the eccentricity of a particle.
    :param mass:
    :param velocity:
    :param mass_grav_body:
    :param position:
    :param orbital_energy:
    :param angular_momentum:
    :return:
    """
    alpha = - G * mass * mass_grav_body
    reduced_mass = (mass * mass_grav_body) / (mass + mass_grav_body)
    return np.sqrt(1 + ((2 * orbital_energy * (angular_momentum ** 2)) / (reduced_mass * alpha ** 2)))


def semi_major_axis(orbital_energy: float, mass: float, mass_grav_body: float):
    """
    Returns the semi-major axis of a particle.
    :param orbital_energy:
    :param mass:
    :param mass_grav_body:
    :return:
    """
    return - (G * mass_grav_body) / (2 * orbital_energy / mass)


def inclination(angular_momentum_vector: np.array, angular_momentum: float):
    """
    Returns the inclination of a particle.
    :param angular_momentum_vector:
    :param angular_momentum:
    :return:
    """
    return np.arccos(angular_momentum_vector[2] / angular_momentum) * (180 / np.pi)


def periapsis(semi_major_axis: float, eccentricity: float):
    """
    Returns the periapsis of a particle.
    :param semi_major_axis:
    :param eccentricity:
    :return:
    """
    return semi_major_axis * (1 - eccentricity)

def equivalent_circular_semi_major_axis(angular_momentum: float, mass: float, mass_grav_body: float):
    """
    Many SPH studies use the equivalent circular semi-major axis to determine if a particle is in the disk.
    The equivalent circular orbit is a circularized (i.e., no eccentricity) orbit with the same angular momentum as
    the non-circularized orbit. This function returns the equivalent circular semi-major axis of a particle.
    :param angular_momentum:
    :param mass:
    :param mass_grav_body:
    :return:
    """
    specific_anguluar_momentum = angular_momentum / mass
    return specific_anguluar_momentum ** 2 / (G * mass_grav_body)
