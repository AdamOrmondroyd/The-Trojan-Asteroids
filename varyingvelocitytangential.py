"""
Generates plot for wander for velocity perturbations tangent to the planet's
orbital circle.
Note that v is positive going away from the planet, towards L3.
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

v_spread = 0.05
points = 100
vs = np.linspace(-v_spread, v_spread, points)


def wander_wrapper(v_offset):
    """Wrapper to put the correct arguments into wander method for given tangential velocity perturbation"""
    return ast.wander(
        ts,
        r_0=ast.L4,
        v_0=np.array([-ast.L4[1], ast.L4[0], 0]) / np.linalg.norm(ast.L4) * v_offset,
        stability_point=ast.L4,
    )


if __name__ == "__main__": # Required for multiprocessing to work properly
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    wanders = pool.map(wander_wrapper, vs)
    pool.close()

    ### Fitting quadratics to both directions of perturbation ###

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    (a1, b1, c1), pcov1 = curve_fit(
        quadratic, vs[points // 2 :], wanders[points // 2 :]
    )

    positive_equation_string = "{:.1f}v$^2$ {:+.1f}v {:+.3e}".format(a1, b1, c1)

    (a2, b2, c2), pcov2 = curve_fit(
        quadratic, vs[: points // 2], wanders[: points // 2]
    )

    negative_equation_string = "{:.1f}v$^2$ {:+.1f}v {:+.3e}".format(a2, b2, c2)

    ### Plotting ###
    
    fig, ax = plt.subplots()

    ax.plot(vs, wanders, label="wanders", marker="+", color="c", linestyle="None")

    ax.plot(
        vs[points // 2 :],
        quadratic(vs[points // 2 :], a1, b1, c1),
        label=positive_equation_string,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.plot(
        vs[: points // 2],
        quadratic(vs[: points // 2], a2, b2, c2),
        label=negative_equation_string,
        color="blue",
        linestyle="--",
        linewidth=1,
    )

    ax.set(
        title="Wander due to velocity perturbation perpendicular to L$_4$",
        xlabel="velocity perturbation perpendicular to L$_4$ / (au/year)",
        ylabel="wander / au",
    )
    ax.legend()

    filename = "plots\\velocity_wanders_perpendicular_L4"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
