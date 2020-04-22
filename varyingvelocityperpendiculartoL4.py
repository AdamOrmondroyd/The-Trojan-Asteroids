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


def max_wander_wrapper(v_offset):
    return max_wander(
        ts,
        r_0=L4,
        v_0=np.array([-L4[1], L4[0], 0]) / np.linalg.norm(L4) * v_offset,
        stability_point=L4,
    )


spread = 0.06
points = 100
vs = np.linspace(0, spread, points)

tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = pool.map(max_wander_wrapper, vs)
    pool.close()

    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    (a, b, c), pcov = curve_fit(quadratic, vs, wanders)

    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))

    fig, ax = plt.subplots()

    ax.plot(vs, wanders, label="wanders", marker="+", linestyle="None")

    ax.plot(vs, quadratic(vs, a, b, c), label="quadratic fit")

    ax.set(
        title="Wander due to velocity perturbation perpendicular to L$_4$",
        xlabel="Velocity perpendicular to L$_4$ / (AU/year)",
        ylabel="Maximum wander / AU",
    )
    ax.legend()
    plt.savefig("velocity_wanders_perpendicular_L4.png")
    plt.show()
