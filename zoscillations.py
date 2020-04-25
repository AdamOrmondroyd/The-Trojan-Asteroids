import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import time
import multiprocessing

ast = RotatingAsteroid()

end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

z_min = 0.01
z_max = 1.0
points = 100
zs = np.linspace(z_min, z_max, points)


def zero(t, y):
    """Event to find when the asteroid passes through the orbital plane"""
    return y[2]


def time_period_wrapper(z):
    sol = ast.trajectory(
        ts, ast.L4 + np.array([0, 0, z]), v_0=np.array([0, 0, 0]), events=zero
    )
    return np.mean(np.diff(sol.t_events)) * 2.0


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    periods = pool.map(time_period_wrapper, zs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    print("Mean time period: {:.4f} ± {:.4f}".format(np.mean(periods), np.std(periods)))

    print("Planet orbit period: {:.4f} years".format(ast.T))

    fig, ax = plt.subplots()

    ax.plot(zs, periods, label="periods", marker="+", color="c", linestyle="None")
    ax.axhline(np.mean(periods), label="mean period", color="k", linestyle="--")
    ax.set(
        title="Wander due to position purturbation along z",
        xlabel="z offset / au",
        ylabel="oscillation period / years",
    )
    ax.legend()

    filename = "plots\\z_oscillations"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
