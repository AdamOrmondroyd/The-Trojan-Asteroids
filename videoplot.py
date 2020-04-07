import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from stationaryframe import asteroid, r_sun, r_j, l_4, l_5
from constants import L4, L5, R, R_SUN, R_J, T, W

fps = 30

end_time = 100
points_per_year = 30
ts = np.linspace(0, end_time, int(end_time * points_per_year))

greek_sol = asteroid(
    run_time=end_time, t_eval=ts, r_0=L4, v_0=np.array([-W * L4[1], W * L4[0], 0])
)
trojan_sol = asteroid(
    run_time=end_time, t_eval=ts, r_0=L5, v_0=np.array([-W * L5[1], W * L5[0], 0])
)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6), aspect="equal")

(greek_line,) = ax.plot(
    [], [], label="Greeks", color="green", marker="o", markersize=5, linestyle="None"
)
(trojan_line,) = ax.plot(
    [], [], label="Trojans", color="magenta", marker="o", markersize=5, linestyle="None"
)
(sun_line,) = ax.plot(
    [], [], label="sun", color="orange", marker="*", markersize=20, linestyle="None"
)
(j_line,) = ax.plot(
    [], [], label="Jupiter", color="red", marker="o", markersize=10, linestyle="None"
)
(l4_line,) = ax.plot(
    [], [], label="L$_4$", color="blue", marker="+", markersize=5, linestyle="None"
)
(l5_line,) = ax.plot(
    [], [], label="L$_4$", color="red", marker="+", markersize=5, linestyle="None"
)

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

ax.set(title="Stationary frame", xlabel="x/AU", ylabel="y/AU")

ax.legend(loc="upper right", frameon=False, prop={"size": 10})


def animate(i):
    greek_line.set_data([greek_sol.y[0, i]], [greek_sol.y[1, i]])
    trojan_line.set_data([trojan_sol.y[0, i]], [trojan_sol.y[1, i]])
    sun_position = r_sun(greek_sol.t[i])
    sun_line.set_data(sun_position[0], sun_position[1])
    j_position = r_j(greek_sol.t[i])
    j_line.set_data(j_position[0], j_position[1])
    l4 = l_4(greek_sol.t[i])
    l4_line.set_data(l4[0], l4[1])
    l5 = l_5(greek_sol.t[i])
    l5_line.set_data(l5[0], l5[1])

    time_text.set_text(str(np.round(greek_sol.t[i], 3)) + " years")
    return (greek_line, trojan_line, sun_line, j_line, l4_line, l5_line, time_text)


anim = FuncAnimation(
    fig,
    animate,
    frames=end_time * points_per_year,  # supplies range(frames) to animate
    interval=1 / fps,  # time between frames
    blit=True,
)
print("animated")

writer = animation.FFMpegWriter(
    fps=fps, metadata=dict(artist="Adam Ormondroyd"),  # bitrate=1800
)
print("writer gotten")
anim.save("movie.mp4", writer=writer)
print("saved")
plt.show()
