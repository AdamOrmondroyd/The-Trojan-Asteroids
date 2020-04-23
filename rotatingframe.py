import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import G, M_SUN, M_J, R, R_SUN, R_J, W

r_sun = np.array([-R_SUN, 0, 0])
r_j = np.array([R_J, 0, 0])


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


def asteroid(t_eval, r_0, v_0):
    """Trajectory of asteroid calculated using solve_ivp"""
    y0 = np.append(r_0, v_0)

    return solve_ivp(derivs, (0, t_eval[-1]), y0, t_eval=t_eval, method="LSODA")


def max_wander(t_eval, r_0, v_0, stability_point):
    """Find the maximum distance from the starting point for given initial conditions in the rotating frame"""
    sol = asteroid(t_eval, r_0, v_0)
    rs = sol.y[0:3]  # extract positions from solution
    deltas = (rs.T - stability_point).T  # Transpose used to stick to array convention
    norms = np.linalg.norm(deltas, axis=0)
    return norms.max()
