from numpy import pi

M_SUN = 1
M_J = 0.001
R = 5.2
R_SUN = R * M_J / (M_J + M_SUN)
R_J = R * M_SUN / (M_J + M_SUN)
W = 2 * pi * (M_J + M_SUN) ** (1 / 2) / R ** (3 / 2)
