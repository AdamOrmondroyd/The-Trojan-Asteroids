import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from constants import L4, L5, M_SUN, M_J, R, W

R_SUN = R * M_J / (M_J + M_SUN)
R_J = R * M_SUN / (M_J + M_SUN)
G = 4 * np.pi ** 2


def r_sun(t):
    """Position of the sun at time t"""
    return np.array([-R_SUN * np.cos(W * t), -R_SUN * np.sin(W * t), 0])


def r_j(t):
    """Position of Jupiter at time t"""
    return np.array([R_J * np.cos(W * t), R_J * np.sin(W * t), 0])


def l_4(t):
    """Position of L_4 at time t"""
    return np.array(
        [
            L4[0] * np.cos(W * t) - L4[1] * np.sin(W * t),
            L4[0] * np.sin(W * t) + L4[1] * np.cos(W * t),
            0,
        ]
    )


def l_5(t):
    """Position of L_4 at time t"""
    return np.array(
        [
            L5[0] * np.cos(W * t) - L5[1] * np.sin(W * t),
            L5[0] * np.sin(W * t) + L5[1] * np.cos(W * t),
            0,
        ]
    )


def acceleration(t, r):
    """Acceleration of an asteroid at position r and time t"""
    return -G * (
        M_SUN * (r - r_sun(t)) / np.linalg.norm(r - r_sun(t)) ** 3
        + M_J * (r - r_j(t)) / np.linalg.norm(r - r_j(t)) ** 3
    )


def f(t, y):
    """Position and velocity of asteroid for solver"""
    return np.hstack((y[3:6], acceleration(t, y[0:3])))


def asteroid(run_time, t_eval, r_0, v_0):
    """Trajectory of asteroid calculated using solve_ivp"""
    y0 = np.append(r_0, v_0)
    return solve_ivp(f, (0, run_time), y0, t_eval=t_eval, method="Radau")
