#!/usr/bin/env python3
import pandas as pd
from random import randint

from src.combine import CombinedFile
from src.identify import ParticleMap

runs = [
    {
        "name": "A",
        "path": "/home/theia/scotthull/Paper3_SPH/gi/mars_citron_1.4vesc_b073_stewart_undiff",
        "num_processes": 400,
    },
]
