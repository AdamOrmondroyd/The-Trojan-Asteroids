"""
Generates plots investigating oscillations in the z direction

Uses the rotating frame to plot the oscillating behaviour of an asteroid perturbed
in the z direction, then plots time period for varying perturbations.
"""
import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import multiprocessing

ast = RotatingAsteroid()

# 100 samples per year for 100 planetary orbits
end_time = 10 * ast.T
points_per_year = 100
ts = np.linspace(0, end_time, int(end_time * points_per_year))

z_min = 0.01
z_max = 1.0
points = 50
zs = np.linspace(z_min, z_max, points)


def zero(t, y):
    """Event to find when the asteroid passes through the orbital plane"""
    return y[2]


def time_period_wrapper(z):
    """Wrapper to find time period for given z position perturbation"""
    sol = ast.trajectory(
        ts, ast.L4 + np.array([0, 0, z]), v_0=np.array([0, 0, 0]), events=zero
    )
    return np.mean(np.diff(sol.t_events)) * 2.0


### Single plot for demonstration ###

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

z = 1.0

sol = ast.trajectory(ts, ast.L4 + np.array([0, 0, z]), v_0=np.array([0, 0, 0]))

### z against time ###

ax[0].plot(ts, sol.y[2], label="oscillations", color="k", linestyle="-")

ax[0].set(
    title="z oscillations: time period = {:.3f} years".format(time_period_wrapper(z)),
    xlabel="time / years",
    ylabel="z / au",
)

### Motion in orbital plane ###

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

if __name__ == "__main__":  # Required for multiprocessing to work properly

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    periods = pool.map(time_period_wrapper, zs)
    pool.close()

    expected_T = ast.T * np.sqrt(ast.R / np.linalg.norm(ast.L4))

    print("Planet orbit period: {:.4f} years".format(ast.T))
    print("Predicted period for small oscillations: {:.4f} years".format(expected_T))

    ax[2].plot(zs, periods, label="periods", marker="+", color="c", linestyle="None")
    ax[2].axhline(
        ast.T,
        label="predicted period = {:.3f} years".format(expected_T),
        color="k",
        linestyle="--",
    )
    ax[2].set(
        title="Time periods",
        xlabel="z offset / au",
        ylabel="oscillation period / years",
    )
    ax[2].legend(frameon=False)

    fig.tight_layout()

    # filename = "plots\\z_oscillations"
    # plt.savefig(filename + ".png")
    # plt.savefig(filename + ".eps")
    plt.show()
