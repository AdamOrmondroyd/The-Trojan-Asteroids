import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stationaryframe import asteroid, r_sun, r_j
from constants import R, R_SUN, R_J

end_time = 100
points_per_year = 1000
ts = np.linspace(0, end_time, int(end_time * points_per_year))

sol = asteroid(run_time=end_time, t_eval=ts)

# plt.plot(-R_SUN, 0, "+", label="sun", color="orange")
# plt.plot(R_J, 0, "+", label="Jupiter", color="red")
# plt.plot(sol.y[0], sol.y[1], "-", label="Greeks", color="green")
# plt.axis("scaled")
# plt.legend()
# plt.show()

fig = plt.figure()
ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6), aspect="equal")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
(trojan_line,) = ax.plot([], [], "g+")
(sun_line,) = ax.plot([], [], "y+")
(j_line,) = ax.plot([], [], "r+")


def animate(i):
    trojan_line.set_data([sol.y[0, i]], [sol.y[1, i]])
    sun_position = r_sun(sol.t[i])
    sun_line.set_data(sun_position[0], sun_position[1])
    j_position = r_j(sol.t[i])
    j_line.set_data(j_position[0], j_position[1])
    time_text.set_text(np.round(sol.t[i], 3) + " years")
    return (trojan_line, sun_line, j_line, time_text)


anim = FuncAnimation(
    fig,
    animate,
    frames=end_time * points_per_year,
    interval=int(1 / points_per_year),
    blit=True,
)
plt.show()
