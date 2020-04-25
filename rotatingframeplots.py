import numpy as np
import matplotlib.pyplot as plt
from constants import M_SUN
from rotatingframe import RotatingAsteroid

ast = RotatingAsteroid()

points_per_year = 1000

# Approximate radius of L3
L3 = ast.R_P + 7 * ast.R * ast.M_P / (12 * M_SUN)

fig, ax = plt.subplots(2, 2, figsize=(7, 8))

### Tadpole ###

end_time = 15 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

tadpole_sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.001, v_0=np.array([0, 0, 0]))

ax[0, 0].plot(
    tadpole_sol.y[0], tadpole_sol.y[1], label="asteroid", color="green", linestyle="-"
)
ax[0, 0].plot(ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+")
ax[0, 0].set(
    title="Tadpole", xlabel="x/au", ylabel="y/au", xlim=[2, 3], ylim=[4, 5],
)
ax[0, 0].set_aspect("equal", "box")

### Curved tadpole ###

end_time = 15 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

curved_sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.01, v_0=np.array([0, 0, 0]))

ax[0, 1].plot(
    -ast.R_SUN,
    0,
    label="Sun",
    color="orange",
    marker="*",
    markersize=20,
    linestyle="None",
)
ax[0, 1].plot(
    ast.R_P,
    0,
    label="Jupiter",
    color="red",
    marker="o",
    markersize=10,
    linestyle="None",
)
ax[0, 1].plot(
    curved_sol.y[0], curved_sol.y[1], label="asteroid", color="green", linestyle="-"
)
ax[0, 1].plot(
    -L3, 0, label="L$_3$", color="k", marker="+", linestyle="None",
)
ax[0, 1].plot(
    ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+", linestyle="None"
)
ax[0, 1].plot(
    ast.L5[0], ast.L5[1], label="L$_5$", color="red", marker="+", linestyle="None"
)
ax[0, 1].set(
    title="Curved tadpole", xlabel="x/au", ylabel="y/au", xlim=[-6, 6], ylim=[-6, 6],
)
ax[0, 1].set_aspect("equal", "box")

### Horseshoe ###

end_time = 30 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

horseshoe_sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.02, v_0=np.array([0, 0, 0]))

ax[1, 0].plot(
    -ast.R_SUN,
    0,
    label="Sun",
    color="orange",
    marker="*",
    markersize=20,
    linestyle="None",
)
ax[1, 0].plot(
    ast.R_P,
    0,
    label="Jupiter",
    color="red",
    marker="o",
    markersize=10,
    linestyle="None",
)
ax[1, 0].plot(
    horseshoe_sol.y[0],
    horseshoe_sol.y[1],
    label="Asteroid",
    color="green",
    linestyle="-",
)
ax[1, 0].plot(
    -L3, 0, label="L$_3$", color="k", marker="+", linestyle="None",
)
ax[1, 0].plot(
    ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+", linestyle="None"
)
ax[1, 0].plot(
    ast.L5[0], ast.L5[1], label="L$_5$", color="red", marker="+", linestyle="None"
)
ax[1, 0].set(
    title="Horseshoe", xlabel="x/au", ylabel="y/au", xlim=[-6, 6], ylim=[-6, 6],
)
ax[1, 0].set_aspect("equal", "box")

### Passing ###

end_time = 15 * ast.T
ts = np.linspace(0, end_time, int(end_time * points_per_year))

passing_sol = ast.trajectory(t_eval=ts, r_0=ast.L4 * 1.045, v_0=np.array([0, 0, 0]))

ax[1, 1].plot(
    -ast.R_SUN,
    0,
    label="Sun",
    color="orange",
    marker="*",
    markersize=20,
    linestyle="None",
)
ax[1, 1].plot(
    ast.R_P,
    0,
    label="Jupiter",
    color="red",
    marker="o",
    markersize=10,
    linestyle="None",
)
ax[1, 1].plot(
    passing_sol.y[0], passing_sol.y[1], label="asteroid", color="green", linestyle="-"
)
ax[1, 1].plot(
    -L3, 0, label="L$_3$", color="k", marker="+", linestyle="None",
)
ax[1, 1].plot(
    ast.L4[0], ast.L4[1], label="L$_4$", color="blue", marker="+", linestyle="None"
)
ax[1, 1].plot(
    ast.L5[0], ast.L5[1], label="L$_5$", color="red", marker="+", linestyle="None"
)
ax[1, 1].set(
    title="Passing",
    xlabel="x/au",
    ylabel="y/au",
    xlim=[-8, 8],
    ylim=[-8, 8],
    xticks=np.linspace(-8, 8, 9),
)
ax[1, 1].set_aspect("equal", "box")


fig.tight_layout()

handles, labels = ax[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="center", ncol=len(labels))

filename = "plots\\rotating_frame_plots"
plt.savefig(filename + ".png")
plt.savefig(filename + ".eps")
plt.show()
