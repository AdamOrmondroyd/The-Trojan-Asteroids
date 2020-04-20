import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import max_wander
from constants import L4, L5, R, R_SUN, R_J, T
import time
import multiprocessing
from itertools import product
from scipy.optimize import curve_fit

end_time = 100 * T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))


def max_wander_wrapper(z_offset):
    return max_wander(
        end_time,
        ts,
        r_0=L4 + np.array([0, 0, z_offset]),
        v_0=np.array([0, 0, 0]),
        stability_point=L4,
    )


spread = 1.0
points = 100
zs = np.linspace(0, spread, points)

tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = pool.map(max_wander_wrapper, zs)
    pool.close()

    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    def linear(x, a, b):
        return a * x + b

    (a, b, c), pcov = curve_fit(quadratic, zs, wanders)

    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))

    (d, e), pcov2 = curve_fit(linear, zs, wanders)

    print("d: " + str(d))
    print("e: " + str(e))

    fig, ax = plt.subplots()

    ax.plot(zs, wanders, marker="+", label="wanders")

    ax.plot(zs, quadratic(zs, a, b, c), label="quadratic fit")

    ax.plot(zs, linear(zs, d, e), label="linear fit")

    ax.set(
        title="Wander due to position purturbation along z",
        xlabel="z offset / AU",
        ylabel="Maximum wander / AU",
    )
    ax.legend()
    plt.savefig("position_wanders_z.png")
    plt.show()
