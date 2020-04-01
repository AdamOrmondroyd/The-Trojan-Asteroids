import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M_SUN = 1
M_J = 0.001
R = 5.2
R_SUN = R * M_J / (M_J + M_SUN)
R_J = R * M_SUN / (M_J + M_SUN)
W = 2 * np.pi * (M_J + M_SUN) ** (1 / 2) / R ** (3 / 2)  # angular velocity
G = 4 * np.pi ** 2


def r_sun(t):
    """Position of the sun at time t"""
    return np.array([-R_SUN * np.cos(W * t), -R_SUN * np.sin(W * t), 0])


def r_j(t):
    """Position of Jupiter at time t"""
    return np.array([R_J * np.cos(W * t), R_J * np.sin(W * t), 0])


def acceleration(t, r):
    """Acceleration of an asteroid at position r and time t"""
    return -G * (
        M_SUN * (r - r_sun(t)) / np.linalg.norm(r - r_sun(t)) ** 3
        + M_J * (r - r_j(t)) / np.linalg.norm(r - r_j(t)) ** 3
    )


def f(t, y):
    """Position and velocity of asteroid for solver"""
    return np.hstack((y[3:6], acceleration(t, y[0:3])))


def asteroid(run_time, t_eval):
    """Trajectory of asteroid calculated using solve_ivp"""
    theta0 = np.pi / 3
    y0 = np.array(
        [
            R * np.cos(theta0) - R_SUN,
            R * np.sin(theta0),
            0,
            -W * R * np.sin(theta0),
            W * (R * np.cos(theta0)),
            0,
        ]
    )

    return solve_ivp(f, (0, run_time), y0, t_eval=t_eval, method="LSODA")


end_time = 100
points_per_year = 1000

ts = np.linspace(0, end_time, end_time * points_per_year)

r_sunxs, r_sunys, dump = r_sun(ts)
r_jxs, r_jys, dump = r_j(ts)

sol = asteroid(run_time=end_time, t_eval=ts)
print(sol.y)

plt.plot(r_sunxs, r_sunys, label="sun", color="orange")
plt.plot(r_jxs, r_jys, label="Jupiter", color="red")
plt.plot(sol.y[0], sol.y[1], label="Greeks", color="green")
plt.axis("scaled")
plt.legend()
plt.show()
