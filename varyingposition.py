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


tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = np.reshape(
        pool.starmap(max_wander_wrapper, product(xs, xs)), (points, points), order="F"
    )
    pool.close()

    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    xx, yy = np.meshgrid(xs, xs)

    fig, ax = plt.subplots()

    contours = ax.contourf(xx, yy, wanders, levels=100)
    cbar = fig.colorbar(contours)
    cbar.set_label("Wander / au")

    ax.plot(0, 0, marker="+", label="L%_4$")

    rs = np.outer(ast.L4, xs) / np.linalg.norm(ast.L4)
    ax.plot(rs[0], rs[1], label="Positions for next part")

    planet_circle = plt.Circle((-ast.L4[0], -ast.L4[1]), ast.R_P, color="r", fill=False)
    ax.add_artist(planet_circle)

    ax.set_aspect("equal", "box")
    ax.set(
        title="Wander as a function of initial position",
        xlabel="x offset / au",
        ylabel="y_offset / au",
    )
    plt.savefig("position_wanders.png")
    plt.show()
