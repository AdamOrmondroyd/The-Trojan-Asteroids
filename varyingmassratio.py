import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import asteroid
import time
import multiprocessing
from scipy.optimize import curve_fit


### Starts to die at around m=0.36
m_min = 0.001
m_max = 0.025
points = 100
ms = np.linspace(m_min, m_max, points)


def max_wander_wrapper(m):
    ast = asteroid(M_J=m)
    print(ast.M_J)
    end_time = 100 * ast.T
    points_per_year = 100
    ts = np.linspace(0, end_time, int(end_time * points_per_year))
    return ast.max_wander(
        ts, r_0=ast.L4, v_0=np.array([0, 0, 0]), stability_point=ast.L4,
    )


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=1)
    wanders = pool.map(max_wander_wrapper, ms)
    pool.close()

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
        xlabel="M$_\textJ$/M$_{\odot}}$",
        ylabel="Maximum wander / au",
    )
    ax.legend()
    plt.savefig("mass_wanders.png")
    plt.show()
