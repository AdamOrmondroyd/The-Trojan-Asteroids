import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import asteroid
import time
import multiprocessing
from scipy.optimize import curve_fit

ast = asteroid()

end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))


def max_wander_wrapper(r_offset):
    return ast.max_wander(
        ts,
        r_0=ast.L4 * (1.0 + r_offset / np.linalg.norm(ast.L4)),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


spread = 0.03
points = 100
rs = np.linspace(-spread, 0, points)

tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = pool.map(max_wander_wrapper, rs)
    pool.close()

    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    def linear(x, a, b):
        return a * x + b

    (a, b, c), pcov = curve_fit(quadratic, rs, wanders)

    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))

    (d, e), pcov2 = curve_fit(linear, rs, wanders)

    print("d: " + str(d))
    print("e: " + str(e))

    fig, ax = plt.subplots()

    ax.plot(rs, wanders, label="wanders", marker="+", linestyle="None")

    ax.plot(rs, quadratic(rs, a, b, c), label="quadratic fit")

    ax.plot(rs, linear(rs, d, e), label="linear fit")

    ax.set(
        title="Wander due to position purturbation along L$_4$",
        xlabel="Offset along L$_4$ / au",
        ylabel="Maximum wander / au",
    )
    ax.legend()
    plt.savefig("position_wanders_along_L4.png")
    plt.show()
