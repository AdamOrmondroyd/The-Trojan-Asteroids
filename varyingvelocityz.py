"""
Generates plot for wander for velocity perturbations in the z direction from L4.
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

vz_spread = 0.5
points = 100
vs = np.linspace(-vz_spread, vz_spread, points)


def wander_wrapper(v_offset):
    """Wrapper to put correct arguments into wander method for given z velocity perturbation"""
    return ast.wander(
        ts, r_0=ast.L4, v_0=np.array([0, 0, v_offset]), stability_point=ast.L4,
    )


if __name__ == "__main__":  # Required for multiprocessing to work properly
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    wanders = pool.map(wander_wrapper, vs)
    pool.close()

    ### Fitting quadratic ###

    def quadratic(x, a, b):
        return a * x ** 2 + b

    (a, b), pcov = curve_fit(quadratic, vs, wanders)

    equation_string = "{:.2f}v$_z^2$ {:+.2e}".format(a, b)

    ### Plotting ###

    fig, ax = plt.subplots()

    ax.plot(vs, wanders, label="wanders", color="c", marker="+", linestyle="None")
    ax.plot(vs, quadratic(vs, a, b), label=equation_string, color="k", linestyle="--")

    ax.set(
        title="Wander due to velocity perturbation along z",
        xlabel="z velocity perturbation / (au/year)",
        ylabel="wander / au",
    )
    ax.legend()

    filename = "plots\\velocity_wanders_z"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
