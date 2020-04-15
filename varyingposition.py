import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import max_wander
from constants import L4, L5, R, R_SUN, R_J, T
import time
import multiprocessing
import concurrent.futures

end_time = 100 * T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))


def y(y):
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


spread = 0.04
points = 16
xs = np.linspace(-spread, spread, points)
tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = pool.map(y, xs)
    np.save("wanders.npy", (wanders, spread, xs), allow_pickle=True)
    print("saved")
    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    wanders, spread, xs = np.load("wanders.npy", allow_pickle=True)

    fig, axis = plt.subplots()

    xx, yy = np.meshgrid(xs, xs)
    contours = axis.contourf(xx, yy, wanders)
    cbar = fig.colorbar(contours)
    cbar.set_label("Wander / AU")
    plt.savefig("position_wanders.png")
    plt.show()
