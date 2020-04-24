import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import rotatingframe
from scipy.optimize import curve_fit

### Starts to die at around m=0.36
m_min = 0.001
m_max = 0.025
points = 100
ms = np.linspace(m_min, m_max, points)
wanders = np.zeros(points)


for i in range(points):
    print(i)
    rotatingframe.M_J = ms[i]
    rotatingframe.R = 5.2
    rotatingframe.R_SUN = rotatingframe.R * ms[i] / (ms[i] + rotatingframe.M_SUN)
    rotatingframe.R_J = (
        rotatingframe.R * rotatingframe.M_SUN / (ms[i] + rotatingframe.M_SUN)
    )
    rotatingframe.W = (
        2 * pi * (ms[i] + rotatingframe.M_SUN) ** (1 / 2) / rotatingframe.R ** (3 / 2)
    )
    rotatingframe.T = 2 * pi / rotatingframe.W

    rotatingframe.L4 = np.array(
        [rotatingframe.R / 2 - rotatingframe.R_SUN, rotatingframe.R * np.sqrt(3) / 2, 0]
    )
    rotatingframe.L5 = np.array(
        [
            rotatingframe.R / 2 - rotatingframe.R_SUN,
            -rotatingframe.R * np.sqrt(3) / 2,
            0,
        ]
    )

    rotatingframe.r_sun = np.array([-rotatingframe.R_SUN, 0, 0])
    rotatingframe.r_j = np.array([rotatingframe.R_J, 0, 0])

    print(rotatingframe.M_J)

    end_time = 100 * rotatingframe.T
    points_per_year = 100
    ts = np.linspace(0, end_time, int(end_time * points_per_year))

    wanders[i] = rotatingframe.max_wander(
        ts,
        r_0=rotatingframe.L4,
        v_0=np.array([0, 0, 0]),
        stability_point=rotatingframe.L4,
    )


def quadratic(x, a, b, c):
    """Quadratic for curve fit"""
    return a * x ** 2 + b * x + c


def linear(x, a, b):
    return a * x + b


(a, b, c), pcov = curve_fit(quadratic, ms, wanders)

print("a: " + str(a))
print("b: " + str(b))
print("c: " + str(c))

(d, e), pcov2 = curve_fit(linear, ms, wanders)

print("d: " + str(d))
print("e: " + str(e))

fig, ax = plt.subplots()

ax.plot(ms, wanders, label="wanders", marker="+", linestyle="None")
ax.plot(ms, quadratic(ms, a, b, c), label="quadratic fit")

ax.set(
    title="Varying the mass ratio",
    xlabel="M$_{\mathrm{J}}$/M$_{\odot}$",
    ylabel="Maximum wander / au",
)
ax.legend()
plt.savefig("mass.png")
plt.show()
