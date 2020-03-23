import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M_SUN = 1
M_J = 0.001
R = 5.2
R_SUN = np.array([-R * M_J / (M_J + M_SUN), 0, 0]
R_J = np.array([R * M_SUN / (M_J + M_SUN), 0, 0]
W = 2 * np.pi * (M_J + M_SUN) ** (1 / 2) / R ** (3 / 2)  # angular velocity
G = 4 * np.pi ** 2

def acceleration(t, r, v):
    """Acceleration of an asteroid at position r and time t"""
    return -G * (
        M_SUN * (r - r_sun(t)) / np.linalg.norm(r - r_sun(t)) ** 3
        + M_J * (r - r_j(t)) / np.linalg.norm(r - r_j(t)) ** 3
    ) + W * np.array([2 * v[1]+ W * r[0], -2*v[0]+ W *r[1], 0])