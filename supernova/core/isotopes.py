"""
Small implementation for the ion species mass.
This will read the relevant information from the mendeleev library.
"""

import numpy as np
import matplotlib.pyplot as plt
import unyt

isotopes_mass: dict[str, unyt.unyt_quantity] =  {
    'H': 1.00784 * unyt.amu,
    'D': 2.01410 * unyt.amu,
    'T': 3.01605 * unyt.amu,
    'He3': 3.01603 * unyt.amu,
    'He4': 4.00260 * unyt.amu,
}