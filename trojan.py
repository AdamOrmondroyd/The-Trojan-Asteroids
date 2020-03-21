import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
    """Acceleration of an asteroid at position r and time t"""
    return -G * (
        M_SUN * (r - r_sun(t)) / np.linalg.norm(r - r_sun(t)) ** 3
        + M_J * (r - r_j(t)) / np.linalg.norm(r - r_j(t)) ** 3
    )


def f(t, y):
    """Position of asteroid for solver"""
    return np.hstack((y[3:6], acceleration(t, y[0:3])))


def asteroid(run_time, t_eval, R0=R_J, theta0=np.pi / 3):
    """Trajectory of asteroid calculated using solve_ivp"""
    y0 = np.array(
        [
            R0 * np.cos(theta0),
            R0 * np.sin(theta0),
            0,
            -OMEGA * R0 * np.sin(theta0),
            OMEGA * R0 * np.cos(theta0),
            0,
        ]
    )
    print(y0[0])
    return solve_ivp(f, (0, run_time), y0, t_eval=t_eval)


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
