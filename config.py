from pyscf.scf.diis import CDIIS, EDIIS, ADIIS
from algo import *

# Benchmarks to compare
DIIS_METHODS = [
    CDIIS,
    EDIISCDIIS,
    LCIIS
]

# DIIS history size
DIIS_M = 20

# When to start DIIS (Iteration 0)
DIIS_START = 0

# Grid level for DFT
GRID_LEVEL = 3

# Energy & Gradient Convergence Criteria
CONV_E = 1e-10
CONV_G = 1e-5

# Plot configuration
PLOT_LINE_WIDTH = 1.2
PLOT_MARKER_SIZE = 5
