import numpy as np
import pandas as pd

from src.interpolation import NearestNeighbor1D


class GiantImpactReport:
    """
    This class is responsible for generating a report of the giant impact simulation.
    """

    def calculate_vmf(self, particles: pd.DataFrame, phase_curve: pd.DataFrame):
        """
        Calculates the vapor mass fraction (vmf) of the particles.
        Assumes single phase particles.
        """
        # if the number of particles is zero, return zero
        if len(particles) == 0:
            return [0.0, 0.0]
        nn = NearestNeighbor1D()
        # if the total entropy column is not defined, raise an error stating that it is required
        # ideally, total entropy should include both the particle entropy and the entropy gain associated with the
        # orbital circularization
        if 'total entropy' not in particles.columns:
            raise AttributeError("A 'total entropy' column is required to calculate the velocity mass fraction.\n"
                                 "This is the sum of the particle entropy and the entropy "
                                 "gain associated with orbital circularization.")
        # define the supercritical temperature
        supercritical_T = max(phase_curve['temperature'])
        # get the nearest temperature index for each particle on the phase curve corresponding to the
        # particle's temperature
        # apply the function to each particle in the list
        particles['nearest temperature index'] = particles['temperature'].apply(
            nn.neighbor_index, args=(list(phase_curve['temperature']),))
        # particles['nearest temperature index'] = nn.neighbor_index(given_val=particles['temperature'],
        #                                                             array=list(phase_curve['temperature']))
        # get the corresponding phase curve entropy on the liquid and vapor curves
        # do so for each particle in the array
        particles['nearest liquid entropy'] = particles['nearest temperature index'].apply(
            lambda x: phase_curve['entropy_sol_liq'][x])
        particles['nearest vapor entropy'] = particles['nearest temperature index'].apply(
            lambda x: phase_curve['entropy_vap'][x])
        # particles['nearest liquid entropy'] = phase_curve['entropy_sol_liq'][particles['nearest temperature index']]
        # particles['nearest vapor entropy'] = phase_curve['entropy_vap'][particles['nearest temperature index']]
        # define rules on whether the particle is pure liquid or pure vapor, or supercritical or mixed phase
        is_supercritical = particles['temperature'] >= supercritical_T

        # calculate vmf with effects of orbital circularization
        is_pure_liquid_w_circ = particles['total entropy'] <= particles['nearest liquid entropy']
        is_pure_vapor_w_circ = particles['total entropy'] >= particles['nearest vapor entropy']
        # calculate the vmf for each particle
        particles['vmf_w_circ'] = np.where(
            is_supercritical, 1.0, np.where(
                is_pure_liquid_w_circ, 0.0, np.where(
                    is_pure_vapor_w_circ, 1.0, (particles['total entropy'] - particles['nearest liquid entropy']) / (
                            particles['nearest vapor entropy'] - particles['nearest liquid entropy'])
                )
            )
        )

        is_pure_liquid_wo_circ = particles['entropy'] <= particles['nearest liquid entropy']
        is_pure_vapor_wo_circ = particles['entropy'] >= particles['nearest vapor entropy']
        # calculate vmf without effects of orbital circularization
        particles['vmf_wo_circ'] = np.where(
            is_supercritical, 1.0, np.where(
                is_pure_liquid_wo_circ, 0.0, np.where(
                    is_pure_vapor_wo_circ, 1.0, (particles['entropy'] - particles['nearest liquid entropy']) / (
                            particles['nearest vapor entropy'] - particles['nearest liquid entropy'])
                )
            )
        )

        return [
            particles['vmf_w_circ'].sum() / len(particles) * 100,
            particles['vmf_wo_circ'].sum() / len(particles) * 100
        ]  # return as units of percent

    def generate_report(self, particles: pd.DataFrame, planet_mass_normalizer: float, disk_mass_normalizer: float,
                        vmf_w_circ: float, vmf_wo_circ: float):
        protoplanet_particles = particles[particles['label'] == 'PLANET']
        disk_particles = particles[particles['label'] == 'DISK']
        escaping_particles = particles[particles['label'] == 'ESCAPE']
        d = {
            r"Proto-planet Mass ($M_{\rm P}$)": protoplanet_particles['mass'].sum() / planet_mass_normalizer,
            r"Disk Mass ($M_{\rm L}$)": disk_particles['mass'].sum() / disk_mass_normalizer,
            r"Escaping Mass ($M_{\rm L}$)": escaping_particles['mass'].sum() / disk_mass_normalizer,
            r"$N_{proto-planet}$": len(protoplanet_particles),
            r"$N_{disk}$": len(disk_particles),
            r"$N_{escaping}$": len(escaping_particles),
            r"Disk Iron Mass Fraction (%)": len(disk_particles[disk_particles['tag'] % 2 != 0]) / len(
                disk_particles) * 100,
            r"$L_{\rm disk}$)": disk_particles['angular momentum'].sum(),
            r"$L_{\rm total}$)": particles['angular momentum'].sum(),
            r"Disk VMF (w/ circ.) (%)": vmf_w_circ,
            r"Disk VMF (w/o circ.) (%)": vmf_wo_circ,
            r"Avg. $S_{\rm disk}$ (w/ circ.) (J/kg/K)": disk_particles['total entropy'].mean(),
            r"Avg. $S_{\rm disk}$ (w/o circ.) (J/kg/K)": disk_particles['entropy'].mean(),
            r"Avg. $\Delta S_{\rm circ.}$": np.mean(disk_particles['total entropy'] - disk_particles['entropy']),
            r"Avg. $T_{\rm disk}$ (K)": disk_particles['temperature'].mean(),
            r"Disk Impactor Mass Fraction (%)": len(disk_particles[disk_particles['tag'] > 1]) /
                                                len(disk_particles) * 100,
        }
        return d
