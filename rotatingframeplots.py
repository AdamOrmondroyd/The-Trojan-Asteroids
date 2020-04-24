import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import asteroid

ast = asteroid()

points_per_year = 1000

fig, ax = plt.subplots(1, 3, figsize=(10, 4.3))

### Tadpole ###

end_time = 15 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.001, v_0=np.array([0, 0, 0]))

ax[0].plot(sol.y[0], sol.y[1], label="asteroid", color="green", linestyle="-")
ax[0].plot(ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+")
ax[0].set(
    title="Tadpole", xlabel="x/au", ylabel="y/au", xlim=[2, 3], ylim=[4, 5],
)
ax[0].set_aspect("equal", "box")

### Curved ###

end_time = 15 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.01, v_0=np.array([0, 0, 0]))

ax[1].plot(
    -ast.R_SUN,
    0,
    label="Sun",
    color="orange",
    marker="*",
    markersize=20,
    linestyle="None",
)
ax[1].plot(
    ast.R_P,
    0,
    label="Jupiter",
    color="red",
    marker="o",
    markersize=10,
    linestyle="None",
)
ax[1].plot(sol.y[0], sol.y[1], label="asteroid", color="green", linestyle="-")
ax[1].plot(ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+")
ax[1].plot(ast.L5[0], ast.L5[1], label="L$_4$", color="red", marker="+")
ax[1].set(
    title="Curved", xlabel="x/au", ylabel="y/au", xlim=[-6, 6], ylim=[-6, 6],
)
ax[1].set_aspect("equal", "box")

### Horseshoe ###

end_time = 30 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.02, v_0=np.array([0, 0, 0]))

ax[2].plot(
    -ast.R_SUN,
    0,
    label="Sun",
    color="orange",
    marker="*",
    markersize=20,
    linestyle="None",
)
ax[2].plot(
    ast.R_P,
    0,
    label="Jupiter",
    color="red",
    marker="o",
    markersize=10,
    linestyle="None",
)
ax[2].plot(sol.y[0], sol.y[1], label="Asteroid", color="green", linestyle="-")
ax[2].plot(
    ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+", linestyle="None"
)
ax[2].plot(
    ast.L5[0], ast.L5[1], label="L$_4$", color="red", marker="+", linestyle="None"
)
ax[2].set(
    title="Horseshoe", xlabel="x/au", ylabel="y/au", xlim=[-6, 6], ylim=[-6, 6],
)
ax[2].set_aspect("equal", "box")


fig.tight_layout()

handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels))

filename = "plots\\rotating_frame_plots"
plt.savefig(filename + ".png")
plt.savefig(filename + ".eps")
plt.show()
