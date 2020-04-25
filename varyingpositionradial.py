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
rs = np.linspace(-spread, spread, points)


def wander_wrapper(r_offset):
    return ast.wander(
        ts,
        r_0=ast.L4 * (1.0 + r_offset / np.linalg.norm(ast.L4)),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, rs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    (a1, b1, c1), pcov1 = curve_fit(
        quadratic, rs[points // 2 :], wanders[points // 2 :]
    )

    positive_equation_string = "{:.1f}x$^2$ {:+.1f}x {:+.3e}".format(a1, b1, c1)

    (a2, b2, c2), pcov2 = curve_fit(
        quadratic, rs[: points // 2], wanders[: points // 2]
    )

    negative_equation_string = "{:.1f}x$^2$ {:+.1f}x {:+.3e}".format(a2, b2, c2)

    fig, ax = plt.subplots()

    ax.plot(rs, wanders, label="wanders", marker="+", color="c", linestyle="None")

    ax.plot(
        rs[points // 2 :],
        quadratic(rs[points // 2 :], a1, b1, c1),
        label=positive_equation_string,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.plot(
        rs[: points // 2],
        quadratic(rs[: points // 2], a2, b2, c2),
        label=negative_equation_string,
        color="blue",
        linestyle="--",
        linewidth=1,
    )

    ax.set(
        title="Wander due to position purturbation along L$_4$",
        xlabel="offset along L$_4$ / au",
        ylabel="maximum wander / au",
    )
    ax.legend()

    filename = "plots\\position_wanders_along_L4"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
