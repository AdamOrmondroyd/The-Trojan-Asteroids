import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import max_wander
from constants import L4, L5, R, R_SUN, R_J, T
import time
import multiprocessing
from itertools import product


def y(y, end_time, ts, xs):
    return [
        max_wander(
            end_time,
            ts,
            r_0=L4 + np.array([x, y, 0]),
            v_0=np.array([0, 0, 0]),
            stability_point=L4,
        )
        for x in xs
    ]


def generate_wander(parallel=False):
    """Generating the wanders"""
    end_time = 100 * T
    points_per_year = 100
    ts = np.linspace(0, end_time, int(end_time * points_per_year))

    spread = 0.04
    points = 16
    xs = np.linspace(-spread, spread, points)

    tic = time.process_time()
    if parallel:
        if __name__ == "__main__":

            pool = multiprocessing.Pool(processes=4)  # My processor has 4 cores
            wanders = pool.starmap(y, product(xs, [end_time], [ts], [xs]))
            np.save("wanders.npy", (wanders, spread, xs), allow_pickle=True)
            print("saved")

    else:
        wanders = np.zeros((points, points))
        for j in range(points):
            print(j)
            for i in range(points):
                wanders[j, i] = max_wander(
                    end_time,
                    ts,
                    r_0=L4 + np.array([xs[i], xs[j], 0]),
                    v_0=np.array([0, 0, 0]),
                    stability_point=L4,
                )

        np.save("wanders.npy", (wanders, spread, xs), allow_pickle=True)
    toc = time.process_time()
    print(
        "The average time to generate a pixel was: "
        + str((toc - tic) / points ** 2)
        + "s"
    )


def plot_wanders(save_name=None):
    """Plots the wanders"""
    wanders, spread, xs = np.load("wanders.npy", allow_pickle=True)

    fig, axis = plt.subplots()

    xx, yy = np.meshgrid(xs, xs)
    contours = axis.contourf(xx, yy, wanders)
    cbar = fig.colorbar(contours)
    cbar.set_label("Wander / AU")
    if save_name is not None:
        plt.savefig(save_name + ".png")
    plt.show()


generate_wander(parallel=False)

plot_wanders("position_wanders")
