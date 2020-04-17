import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import max_wander
from constants import L4, L5, R, R_SUN, R_J, T
import time
import multiprocessing
from itertools import product

end_time = 100 * T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))


def max_wander_wrapper(x_offset, y_offset):
    return max_wander(
        end_time,
        ts,
        r_0=L4 + np.array([x_offset, y_offset, 0]),
        v_0=np.array([0, 0, 0]),
        stability_point=L4,
    )


spread = 0.04
points = 32
xs = np.linspace(-spread, spread, points)

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

    fig, axis = plt.subplots()

    contours = axis.contourf(xx, yy, wanders, levels=100)
    cbar = fig.colorbar(contours)
    cbar.set_label("Wander / AU")

    axis.plot(0, 0, "+", label="L%_4$")

    jupiter_circle = plt.Circle((-L4[0], -L4[1]), R_J, color="r", fill=False)
    axis.add_artist(jupiter_circle)

    axis.set(
        title="Wander as a function of initial position",
        xlabel="x offset / AU",
        ylabel="y_offset / AU",
    )
    plt.savefig("position_wanders.png")
    plt.show()
