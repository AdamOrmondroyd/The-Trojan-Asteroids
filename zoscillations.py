import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import time
import multiprocessing

ast = RotatingAsteroid()

end_time = 10 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

z_min = 0.01
z_max = 1.0
points = 100
zs = np.linspace(z_min, z_max, points)


def zero(t, y):
    """Event to find when the asteroid passes through the orbital plane"""
    return y[2]


def time_period_wrapper(z):
    sol = ast.trajectory(
        ts, ast.L4 + np.array([0, 0, z]), v_0=np.array([0, 0, 0]), events=zero
    )
    return np.mean(np.diff(sol.t_events)) * 2.0


### Single plot for demonstration ###

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

z = 1.0

sol = ast.trajectory(ts, ast.L4 + np.array([0, 0, z]), v_0=np.array([0, 0, 0]))

ax[0].plot(ts, sol.y[2], label="oscillations", color="k", linestyle="-")

ax[0].set(
    title="z oscillations: time period = {:.2f} years".format(time_period_wrapper(z)),
    xlabel="z / au",
    ylabel="time / years",
)

ax[1].plot(sol.y[0], sol.y[1], label="asteroid", color="green", linestyle="-")
ax[1].plot(ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+")
ax[1].set(
    title="Resulting tadpole orbit",
    xlabel="x/au",
    ylabel="y/au",
    xlim=[-3, 5],
    ylim=[-1, 7],
)
ax[1].set_aspect("equal")
ax[1].legend()
### Investigating time period ###

if __name__ == "__main__":
    tic = time.time()

    pool = multiprocessing.Pool()
    periods = pool.map(time_period_wrapper, zs)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    print("Planet orbit period: {:.4f} years".format(ast.T))

    ax[2].plot(zs, periods, label="periods", marker="+", color="c", linestyle="None")
    ax[2].axhline(ast.T, label="planet orbit period", color="k", linestyle="--")
    ax[2].set(
        title="Time periods",
        xlabel="z offset / au",
        ylabel="oscillation period / years",
    )
    ax[2].legend()

    fig.tight_layout()

    filename = "plots\\z_oscillations"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
