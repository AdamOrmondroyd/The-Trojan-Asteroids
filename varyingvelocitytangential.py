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

spread = 0.05
points = 100
vs = np.linspace(-spread, spread, points)


def wander_wrapper(v_offset):
    return ast.wander(
        ts,
        r_0=ast.L4,
        v_0=np.array([-ast.L4[1], ast.L4[0], 0]) / np.linalg.norm(ast.L4) * v_offset,
        stability_point=ast.L4,
    )


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, vs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

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
        ylabel="maximum wander / au",
    )
    ax.legend()

    filename = "plots\\velocity_wanders_perpendicular_L4"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
