import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import asteroid
import time
import multiprocessing
from itertools import product

ast = asteroid()

end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

spread = 0.05
points = 32
xs = np.linspace(-spread, spread, points)


def max_wander_wrapper(x_offset, y_offset):
    return ast.max_wander(
        ts,
        r_0=ast.L4 + np.array([x_offset, y_offset, 0]),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = np.reshape(
        pool.starmap(max_wander_wrapper, product(xs, xs)), (points, points), order="F"
    )
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    xx, yy = np.meshgrid(xs, xs)

    fig, ax = plt.subplots()

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
        xlim=[-spread, spread],
        ylim=[-spread, spread],
    )
    ax.legend()

    filename = "plots\\position_wanders"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
