#!/usr/bin/env python3
import pandas as pd
from random import randint

from src.combine import CombinedFile
from src.identify import ParticleMap
from src.report import GiantImpactReport

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
        "name": "D",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/mars_b073_1vesc_rho_c_5kgm3/mars_b073_1vesc_rho_c_5kgm3",
        "num_processes": 600,
        'final_iteration': 219,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "E",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_stewart/mars_b073_1vesc_1.5_rtar/mars_b073_1vesc_1.5_rtar",
        "num_processes": 600,
        'final_iteration': 360,
        'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    },
    {
        "name": "F",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_canup_n_sph/500_mars_b073_1v_esc",
        "num_processes": 600,
        'final_iteration': 360,
        'phase_curve': "src/phase_curves/duniteN_vapour_curve.txt",
    },
    # {
    #     "name": "H",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_stewart/mars_citron_1vesc_b073_stewart_undiff",
    #     "num_processes": 400,
    #     'final_iteration': 1800,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
    # {
    #     "name": "I",
    #     "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_stewart/mars_citron_1.4vesc_b073_stewart_undiff",
    #     "num_processes": 400,
    #     'final_iteration': 1800,
    #     'phase_curve': "src/phase_curves/forstSTS__vapour_curve.txt",
    # },
]

# define the dataframe headers
file_headers = ["id", "tag", "mass", "x", "y", "z", "vx", "vy", "vz", "density", "internal energy", "pressure",
                "potential energy", "entropy", "temperature"]
# define the planet parameters
mass_mars = 6.39e23
equatorial_radius = 3390e3
mass_phobos = 1.0659e16
mass_deimos = 1.4762e15
mass_pd = mass_phobos + mass_deimos

reports = {}  # list of reports

for run in runs:
    # create the combined file
    c = CombinedFile(
        path=run['path'],
        iteration=run['final_iteration'],
        number_of_processes=run['num_processes'],
        to_fname=f"merged_{run['final_iteration']}_{randint(1, int(1e5))}.dat"
    )
    combined_file = c.combine_to_memory()
    # replace the headers
    combined_file.columns = file_headers
    time = c.sim_time
    # create the particle map
    particle_map = ParticleMap(particles=combined_file, mass_planet=mass_mars, equatorial_radius=equatorial_radius)
    particles = particle_map.loop()
    disk_particles = particles[particles['label'] == 'DISK']

    # generate the report
    phase_curve = pd.read_fwf(run['phase_curve'], skiprows=1,
                              names=["temperature", "density_sol_liq", "density_vap", "pressure",
                                     "entropy_sol_liq", "entropy_vap"])
    r = GiantImpactReport()
    vmf_w_circ, vmf_wo_circ = r.calculate_vmf(particles=disk_particles[particles['tag'] % 2 == 0],
                                              phase_curve=phase_curve)
    report = r.generate_report(
        particles=particles,
        planet_mass_normalizer=mass_mars,
        disk_mass_normalizer=mass_pd,
        vmf_w_circ=vmf_w_circ,
        vmf_wo_circ=vmf_wo_circ
    )
    report['time (hrs.)'] = time
    reports[run['name']] = report

# create the dataframe
df = pd.DataFrame(reports)
df.to_csv("gi_report.csv")
