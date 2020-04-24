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

spread = 0.06
points = 100
vs = np.linspace(0, spread, points)


def max_wander_wrapper(v_offset):
    return ast.max_wander(
        ts,
        r_0=ast.L4,
        v_0=np.array([-ast.L4[1], ast.L4[0], 0]) / np.linalg.norm(ast.L4) * v_offset,
        stability_point=ast.L4,
    )


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = pool.map(max_wander_wrapper, vs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

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
        xlabel="Velocity perpendicular to L$_4$ / (au/year)",
        ylabel="Maximum wander / au",
    )
    ax.legend()

    filename = "plots\\velocity_wanders_perpendicular_L4"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
