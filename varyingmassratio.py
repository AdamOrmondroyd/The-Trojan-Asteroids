import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import rotatingframe
from scipy.optimize import curve_fit


m_min = 0.0
m_max = 0.06
points = 100
ms = np.linspace(m_min, m_max, points)
wanders = np.zeros(points)

import constants

for i in range(points):
    print(i)
    constants.M_SUN = 1
    constants.M_J = ms[i]
    constants.R = 5.2
    constants.R_SUN = constants.R * ms[i] / (ms[i] + constants.M_SUN)
    constants.R_J = constants.R * constants.M_SUN / (ms[i] + constants.M_SUN)
    constants.W = 2 * pi * (ms[i] + constants.M_SUN) ** (1 / 2) / constants.R ** (3 / 2)
    constants.T = 2 * pi / constants.W

    constants.L4 = np.array(
        [constants.R / 2 - constants.R_SUN, constants.R * np.sqrt(3) / 2, 0]
    )
    constants.L5 = np.array(
        [constants.R / 2 - constants.R_SUN, -constants.R * np.sqrt(3) / 2, 0]
    )

    from rotatingframe import max_wander

    end_time = 100 * constants.T
    points_per_year = 100
    ts = np.linspace(0, end_time, int(end_time * points_per_year))

    wanders[i] = max_wander(
        end_time,
        ts,
        r_0=constants.L4,
        v_0=np.array([0, 0, 0]),
        stability_point=constants.L4,
    )


def quadratic(x, a, b, c):
    """Quadratic for curve fit"""
    return a * x ** 2 + b * x + c


remove = 5
(a, b, c), pcov = curve_fit(quadratic, ms[remove:], wanders[remove:])

print("a: " + str(a))
print("b: " + str(b))
print("c: " + str(c))

fig, ax = plt.subplots()

ax.plot(ms[remove:], wanders[remove:], label="wanders", marker="+", linestyle="None")
ax.plot(ms[remove:], quadratic(ms[remove:], a, b, c), label="Cubic fit")
plt.show()
