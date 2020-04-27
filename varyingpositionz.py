"""
Generates plot for wander for position perturbations in the z direction from L4.
"""
import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import multiprocessing
from scipy.optimize import curve_fit

ast = RotatingAsteroid()

# 100 samples per year for 100 planetary orbits
end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

z_spread = 1.0
points = 100
zs = np.linspace(-z_spread, z_spread, points)


def wander_wrapper(z_offset):
    """Wrapper to put correct arguments into wander method for given z position perturbation"""
    return ast.wander(
        ts,
        r_0=ast.L4 + np.array([0, 0, z_offset]),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


if __name__ == "__main__":  # Required for multiprocessing to work properly
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    wanders = pool.map(wander_wrapper, zs)
    pool.close()

    ### Fitting quadratic ###

    def quadratic(x, a, b):
        return a * x ** 2 + b

    (a, b), pcov = curve_fit(quadratic, zs, wanders)

    equation_string = "{:.2f}z$^2$ {:+.2e}".format(a, b)

    ### Plotting ###

    fig, ax = plt.subplots()

    ax.plot(zs, wanders, marker="+", label="wanders", linestyle="None", color="c")

    ax.plot(zs, quadratic(zs, a, b), label=equation_string, linestyle="--", color="k")

    ax.set(
        title="Wander due to position purturbation along z",
        xlabel="z offset / au",
        ylabel="wander / au",
    )
    ax.legend()

    filename = "plots\\position_wanders_z"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
