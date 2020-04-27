"""
Generates contour plot for wander for position perturbations in the orbital plane.
"""
import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import multiprocessing
from itertools import product

ast = RotatingAsteroid()

# 100 samples per year for 100 planetary orbits
end_time = 100 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

v_spread = 0.05
points = 32
vs = np.linspace(-v_spread, v_spread, points)


def wander_wrapper(vx_offset, vy_offset):
    return ast.wander(
        ts, r_0=ast.L4, v_0=np.array([vx_offset, vy_offset, 0]), stability_point=ast.L4,
    )


if __name__ == "__main__": # Required for multiprocessing to work properly
    pool = multiprocessing.Pool()
    wanders = np.reshape(
        pool.starmap(wander_wrapper, product(vs, vs)), (points, points), order="F"
    )
    pool.close()

    ### Plotting ###

    fig, ax = plt.subplots()

    xx, yy = np.meshgrid(vs, vs)

    contours = ax.contourf(xx, yy, wanders, levels=100, cmap="viridis_r")
    cbar = fig.colorbar(contours, ticks=np.linspace(0, 11, 12))
    cbar.set_label("wander / au")

    ax.plot(0, 0, label="origin", marker="+", color="blue", linestyle="None")

    v2s = np.outer(np.array([-ast.L4[1], ast.L4[0], 0]), vs) / np.linalg.norm(ast.L4)
    ax.plot(v2s[0], v2s[1], label="velocities to sample", color="k", linewidth=0.5)

    ax.set_aspect("equal", "box")
    ax.set(
        title="Wander as a function of initial velocity",
        xlabel="Initial v$_{\mathrm{x}}$ / (au/year)",
        ylabel="Initial v$_{\mathrm{y}}$ / (au/year)",
        xlim=[-v_spread, v_spread],
        ylim=[-v_spread, v_spread],
    )
    ax.legend()

    filename = "plots\\velocity_wanders"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
