"""
Generates contour plot for wander for position perturbations in the orbital plane.

Uses the rotating frame to calculate the wander of asteroids perturbed from
L4 in the orbital plane, then outputs the results as a contour plot. This file
was used to test the performance of the simulation, so timing has been left in.
"""
import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import multiprocessing
from itertools import product
import time

ast = RotatingAsteroid()

# 100 samples per year for 100 planetary orbits
end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

position_spread = 0.05
points = 32
xs = np.linspace(-position_spread, position_spread, points)


def wander_wrapper(x_offset, y_offset):
    """Wrapper to put correct arguments into wander method for a given position perturbation"""
    return ast.wander(
        ts,
        r_0=ast.L4 + np.array([x_offset, y_offset, 0]),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


if __name__ == "__main__":  # Required for multiprocessing to work properly
    tic = time.time()

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    wanders = np.reshape(
        pool.starmap(wander_wrapper, product(xs, xs)), (points, points), order="F"
    )
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    ### Plotting ###

    fig, ax = plt.subplots()

    xx, yy = np.meshgrid(xs, xs)

    contours = ax.contourf(xx, yy, wanders, levels=100, cmap="viridis_r")
    cbar = fig.colorbar(contours, ticks=np.linspace(0, 11, 12))
    cbar.set_label("wander / au")

    ax.plot(0, 0, label="L$_4$", marker="+", color="blue", linestyle="None")

    rs = np.outer(ast.L4, xs) / np.linalg.norm(ast.L4)
    ax.plot(rs[0], rs[1], label="positions to sample", color="k", linewidth=0.5)

    thetas = np.linspace(0, np.pi / 2, 1000)
    ax.plot(
        ast.R_P * np.cos(thetas) - ast.L4[0],
        ast.R_P * np.sin(thetas) - ast.L4[1],
        label="planet orbit circle",
        color="r",
        linewidth=0.5,
    )

    ax.set_aspect("equal", "box")
    ax.set(
        title="Wander as a function of initial position",
        xlabel="x offset / au",
        ylabel="y offset / au",
        xlim=[-position_spread, position_spread],
        ylim=[-position_spread, position_spread],
    )
    ax.legend()

    # filename = "plots\\position_wanders"
    # plt.savefig(filename + ".png")
    # plt.savefig(filename + ".eps")
    plt.show()
