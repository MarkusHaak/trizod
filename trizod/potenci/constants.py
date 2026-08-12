"""Physical and algorithmic constants for the POTENCI pKa prediction."""

# Physical constants
GAS_CONSTANT = 8.314472  # J/(mol·K)
DIELECTRIC_WATER = 79.0  # relative permittivity of water at ~25°C

# Debye-Hückel charge interaction parameters (Ångströms)
MIN_CHARGE_DISTANCE = 5.0  # minimum distance between titratable groups
DISTANCE_SCALE = 7.5  # scaling factor for sequence-based distances

# Iterative pKa fitting parameters
PKA_WINDOW_HALF = 2  # half-width of sliding window (full window = 2*half + 1 = 5)
PKA_FIT_CYCLES = 5  # number of iterative fitting cycles

# Reference pKa values for titratable residues (including N/C-termini)
REFERENCE_PKA = {
    "n": 8.23,
    "D": 3.86,
    "E": 4.34,
    "H": 6.45,
    "C": 8.49,
    "K": 10.34,
    "R": 13.9,
    "Y": 9.76,
    "c": 3.55,
}
