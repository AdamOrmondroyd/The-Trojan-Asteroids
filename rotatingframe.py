import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import M_SUN, M_J, R, R_SUN, R_J, W

r_sun = np.array([-R_SUN, 0, 0])
r_j = np.array([R_J, 0, 0])
G = 4 * np.pi ** 2


def acceleration(t, r, v):
    """Acceleration of an asteroid at position r and time t"""
    return -G * (
        M_SUN * (r - r_sun) / np.linalg.norm(r - r_sun) ** 3
        + M_J * (r - r_j) / np.linalg.norm(r - r_j) ** 3
    ) + W * np.array(
        [2 * v[1] + W * r[0], -2 * v[0] + W * r[1], 0]
    )  # added coriolis and centrifugal forces


def derivs(t, y):
    """derivatives for solver"""
    return np.hstack((y[3:6], acceleration(t, y[0:3], y[3:6])))


def asteroid(run_time, t_eval, r_0, v_0):
    """Trajectory of asteroid calculated using solve_ivp"""
    theta_0 = np.pi / 3
    y0 = np.array([R * np.cos(theta_0) - R_SUN, R * np.sin(theta_0), 0, 0, 0, 0,])

    return solve_ivp(derivs, (0, run_time), y0, t_eval=t_eval, method="LSODA")
