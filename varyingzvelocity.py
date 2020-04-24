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

spread = 0.6
points = 100
vs = np.linspace(0, spread, points)


def max_wander_wrapper(v_offset):
    return ast.max_wander(
        ts, r_0=ast.L4, v_0=np.array([0, 0, v_offset]), stability_point=ast.L4,
    )


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = pool.map(max_wander_wrapper, vs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    split = 20

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    def linear(x, a, b):
        return a * x + b

    (a, b, c), pcov = curve_fit(quadratic, vs[split:], wanders[split:])

    print("a: " + str(a))
    print("b: " + str(b))
    print("c: " + str(c))

    (d, e), pcov2 = curve_fit(linear, vs[0:split], wanders[0:split])

    print("d: " + str(d))
    print("e: " + str(e))

    fig, ax = plt.subplots()

    ax.plot(vs, wanders, label="wanders", marker="+", linestyle="None")

    ax.plot(vs[split:], quadratic(vs[split:], a, b, c), label="quadratic fit")

    ax.plot(vs[0:split], linear(vs[0:split], d, e), label="linear fit")

    ax.set(
        title="Wander due to velocity perturbation perpendicular along z",
        xlabel="z velocity perturbation / (au/year)",
        ylabel="Maximum wander / au",
    )
    ax.legend()

    filename = "plots\\velocity_wanders_z"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
