import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M_SUN = 1
M_J = 0.001
R = 5.2
R_SUN = np.array([-R * M_J / (M_J + M_SUN), 0, 0])
R_J = np.array([R * M_SUN / (M_J + M_SUN), 0, 0])
W = 2 * np.pi * (M_J + M_SUN) ** (1 / 2) / R ** (3 / 2)  # angular velocity
G = 4 * np.pi ** 2


def acceleration(t, r, v):
    """Acceleration of an asteroid at position r and time t"""
    return -G * (
        M_SUN * (r - R_SUN) / np.linalg.norm(r - R_SUN) ** 3
        + M_J * (r - R_J) / np.linalg.norm(r - R_J) ** 3
    ) + W * np.array(
        [2 * v[1] + W * r[0], -2 * v[0] + W * r[1], 0]
    )  # added coriolis and centrifugal forces


def derivs(t, y):
    """derivatives for solver"""
    return np.hstack((y[3:6], acceleration(t, y[0:3], y[3:6])))


def asteroid(run_time, t_eval):
    """Trajectory of asteroid calculated using solve_ivp"""
    theta0 = np.pi / 3
    y0 = np.array([R * np.cos(theta0) + R_SUN[0], R * np.sin(theta0), 0, 0.01, 0, 0,])

    return solve_ivp(derivs, (0, run_time), y0, t_eval=t_eval, method="LSODA")


end_time = 100
points_per_year = 1000
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = asteroid(run_time=end_time, t_eval=ts)

plt.plot(R_SUN[0], R_SUN[1], "+", label="sun", color="orange")
plt.plot(R_J[0], R_J[1], "+", label="Jupiter", color="red")
plt.plot(sol.y[0], sol.y[1], "-", label="Greeks", color="green")
plt.axis("scaled")
plt.legend()
plt.show()
