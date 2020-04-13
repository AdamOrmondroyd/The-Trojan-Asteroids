import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import max_wander
from constants import L4, L5, R, R_SUN, R_J, T


def generate_wander():
    """Generating the wanders"""
    end_time = 100 * T
    points_per_year = 100
    ts = np.linspace(0, end_time, int(end_time * points_per_year))

    spread = 0.04
    points = 16

    xs = np.linspace(-spread, spread, points)

    wanders = np.zeros((points, points))

    for j in range(points):
        print(j)
        for i in range(points):
            wanders[i, j] = max_wander(
                end_time,
                ts,
                r_0=L4 + np.array([xs[i], xs[j], 0]),
                v_0=np.array([0, 0, 0]),
                stability_point=L4,
            )

    np.save("wanders.npy", (wanders, spread), allow_pickle=True)


def plot_wanders():
    """Plots the wanders"""
    wanders, spread = np.load("wanders.npy", allow_pickle=True)
    fig, axis = plt.subplots()
    im = axis.imshow(wanders, origin="lower", extent=[-spread, spread, -spread, spread])
    fig.colorbar(im)
    plt.show()
    print(wanders)


generate_wander()

plot_wanders()
