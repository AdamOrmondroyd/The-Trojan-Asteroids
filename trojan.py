import numpy as np
import matplotlib.pyplot as plt

M_SUN = 1
M_J = 0.001
R = 5.2
R_SUN = R * M_J / (M_J + M_SUN)
R_J = R * M_SUN / (M_J + M_SUN)
OMEGA = 2 * np.pi * (M_J + M_SUN) / R ** (3 / 2)
G = 4 * np.pi ** 2


def r_sun(t):
    """Position of the sun at time t"""
    return np.array([-R_SUN * np.cos(OMEGA * t), -R_SUN * np.sin(OMEGA * t), 0])


def r_j(t):
    """Position of Jupiter at time t"""
    return np.array([R_J * np.cos(OMEGA * t), R_J * np.sin(OMEGA * t), 0])


def acceleration(t, r):
    return -G * (
        M_SUN * (r - r_sun(t)) / np.linalg.norm(r - r_sun(t)) ** 3
        + M_J * (r - r_j(t)) / np.linalg.norm(r - r_j(t)) ** 3
    )


def asteroid_position(t, y):
    """Position of asteroid"""

    return np.vstack(y[1], acceleration(t, y[0]))


end_time = 10
ts = np.linspace(0, end_time, 100)
print(r_sun(ts))
r_sunxs, r_sunys, dump = r_sun(ts)
r_jxs, r_jys, dump = r_j(ts)

plt.plot(r_sunxs, r_sunys, label="sun")
plt.plot(r_jxs, r_jys, label="Jupiter")
plt.axis("scaled")
plt.show()
