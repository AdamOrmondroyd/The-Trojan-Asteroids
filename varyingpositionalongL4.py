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


def max_wander_wrapper(r_offset):
    return max_wander(
        end_time,
        ts,
        r_0=L4 * (1.0 + r_offset / np.linalg.norm(L4)),
        v_0=np.array([0, 0, 0]),
        stability_point=L4,
    )


spread = 0.06
points = 100
rs = np.linspace(-spread, spread, points)

tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = pool.map(max_wander_wrapper, rs)
    pool.close()

    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    fig, ax = plt.subplots()

    ax.plot(
        rs, wanders,
    )

    ax.set(
        title="Wander due to position purturbation along L$_4$",
        xlabel="Offset along L$_4$ / AU",
        ylabel="Maximum wander / AU",
    )
    plt.savefig("position_wanders_along_L4.png")
    plt.show()
