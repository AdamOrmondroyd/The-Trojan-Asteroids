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


def max_wander_wrapper(vx_offset, vy_offset):
    return ast.max_wander(
        ts, r_0=ast.L4, v_0=np.array([vx_offset, vy_offset, 0]), stability_point=ast.L4,
    )


spread = 0.04
points = 32
vs = np.linspace(-spread, spread, points)
tic = time.time()

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    wanders = np.reshape(
        pool.starmap(max_wander_wrapper, product(vs, vs)), (points, points), order="F"
    )
    pool.close()

    toc = time.time()
    print("Time taken " + str(toc - tic) + "s")

    xx, yy = np.meshgrid(vs, vs)

    fig, ax = plt.subplots()

    contours = ax.contourf(xx, yy, wanders, levels=100)
    cbar = fig.colorbar(contours)
    cbar.set_label("Wander / au")

    ax.plot(0, 0, label="Origin", marker="+")

    v2s = np.outer(np.array([-ast.L4[1], ast.L4[0], 0]), vs) / np.linalg.norm(ast.L4)
    ax.plot(v2s[0], v2s[1], label="Positions for next part")

    jupiter_circle = plt.Circle(
        (-ast.L4[0], -ast.L4[1]), ast.R_J, color="r", fill=False
    )
    ax.add_artist(jupiter_circle)

    ax.set_aspect("equal", "box")
    ax.set(
        title="Wander as a function of initial velocity",
        xlabel="Initial v$_x$ / (au/year)",
        ylabel="Initial v$_y$ / (au/year)",
    )
    plt.savefig("velocity_wanders.png")
    plt.show()
