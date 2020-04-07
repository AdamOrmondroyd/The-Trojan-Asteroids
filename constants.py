import numpy as np
from numpy import pi

M_SUN = 1
M_J = 0.001
R = 5.2
R_SUN = R * M_J / (M_J + M_SUN)
R_J = R * M_SUN / (M_J + M_SUN)
W = 2 * pi * (M_J + M_SUN) ** (1 / 2) / R ** (3 / 2)
T = 2 * pi / W

# Lagrangian points in stationary frame, and at t = 0 in rotating frame
L4 = np.array([R / 2 - R_SUN, R * np.sqrt(3) / 2, 0])
L5 = np.array([R / 2 - R_SUN, -R * np.sqrt(3) / 2, 0])
