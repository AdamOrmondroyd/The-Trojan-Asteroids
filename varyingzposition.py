import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import time
import multiprocessing
from scipy.optimize import curve_fit

ast = RotatingAsteroid()

end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

z_spread = 1.0
points = 100
zs = np.linspace(-z_spread, z_spread, points)


def wander_wrapper(z_offset):
    return ast.wander(
        ts,
        r_0=ast.L4 + np.array([0, 0, z_offset]),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, zs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    (a, b, c), pcov = curve_fit(quadratic, zs, wanders)

    equation_string = "{:.2f}z$^2$ {:+.2f}z {:+.2f}".format(a, b, c)

    fig, ax = plt.subplots()

    ax.plot(zs, wanders, marker="+", label="wanders", linestyle="None", color="c")

    ax.plot(
        zs, quadratic(zs, a, b, c), label=equation_string, linestyle="--", color="k"
    )

    ax.set(
        title="Wander due to position purturbation along z",
        xlabel="z offset / au",
        ylabel="Maximum wander / au",
    )
    ax.legend()

    filename = "plots\\position_wanders_z"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
