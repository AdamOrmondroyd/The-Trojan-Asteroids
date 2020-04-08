import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from stationaryframe import asteroid, r_sun, r_j, l_4, l_5
from constants import L4, L5, R, R_SUN, R_J, T, W


def omega_cross(r):
    """Returns the result of W x r"""
    return np.array([-W * r[1], W * r[0], 0])


def video_plot(
    run_time,
    fps,
    seconds_per_year,
    num_greeks,
    num_trojans,
    position_spread,
    save_animation=False,
    file_name="movie",
):
    """Makes an animation in the stationary frame of a random selection about the Lagrangian points"""

    num_points = int(run_time * fps * seconds_per_year)
    ts = np.linspace(0, run_time, num_points)

    greek_y0s = np.zeros((num_greeks, num_points))
    greek_y1s = np.zeros((num_greeks, num_points))

    for i in range(num_greeks):
        offset = (np.random.rand(3) - 0.5) * position_spread

        offset = offset * np.array([1, 1, 0])  # no z offset

        r_0 = L4 + offset
        v_0 = omega_cross(r_0)
        sol = asteroid(run_time, ts, r_0, v_0)
        greek_y0s[i] = sol.y[0]
        greek_y1s[i] = sol.y[1]

    trojan_y0s = np.zeros((num_trojans, num_points))
    trojan_y1s = np.zeros((num_trojans, num_points))

    for i in range(num_trojans):
        offset = (np.random.rand(3) - 0.5) * position_spread

        offset = offset * np.array([1, 1, 0])  # no z offset

        r_0 = L5 + offset
        v_0 = omega_cross(r_0)
        sol = asteroid(run_time, ts, r_0, v_0)
        trojan_y0s[i] = sol.y[0]
        trojan_y1s[i] = sol.y[1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6), aspect="equal")

    (sun_line,) = ax.plot(
        [], [], label="sun", color="orange", marker="*", markersize=20, linestyle="None"
    )
    (j_line,) = ax.plot(
        [],
        [],
        label="Jupiter",
        color="red",
        marker="o",
        markersize=10,
        linestyle="None",
    )
    (l4_line,) = ax.plot(
        [], [], label="L$_4$", color="blue", marker="+", markersize=10, linestyle="None"
    )
    (l5_line,) = ax.plot(
        [], [], label="L$_5$", color="red", marker="+", markersize=10, linestyle="None"
    )
    (greeks_line,) = ax.plot(
        [],
        [],
        label="Greeks",
        color="green",
        marker="o",
        markersize=1,
        linestyle="None",
    )
    (trojans_line,) = ax.plot(
        [],
        [],
        label="Trojans",
        color="magenta",
        marker="o",
        markersize=1,
        linestyle="None",
    )

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.set(title="Stationary frame", xlabel="x/AU", ylabel="y/AU")

    ax.legend(loc="upper right", frameon=False, prop={"size": 10})

    def animate(i):
        sun_position = r_sun(ts[i])
        sun_line.set_data(sun_position[0], sun_position[1])
        j_position = r_j(ts[i])
        j_line.set_data(j_position[0], j_position[1])

        l4 = l_4(ts[i])
        l4_line.set_data(l4[0], l4[1])
        l5 = l_5(ts[i])
        l5_line.set_data(l5[0], l5[1])

        greeks_line.set_data(greek_y0s[:, i], greek_y1s[:, i])
        trojans_line.set_data(trojan_y0s[:, i], trojan_y1s[:, i])

        time_text.set_text(str(np.round(ts[i], 1)) + " years")
        return (
            sun_line,
            j_line,
            l4_line,
            l5_line,
            greeks_line,
            trojans_line,
            time_text,
        )

    anim = FuncAnimation(
        fig,
        animate,
        frames=int(num_points),  # supplies range(frames) to animate
        interval=1 / fps,  # time between frames
        blit=True,
    )
    print("animated")

    if save_animation:
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Adam Ormondroyd"),  # bitrate=1800
        )

        anim.save(file_name + ".mp4", writer=writer)
        print("saved")
    plt.show()


video_plot(
    run_time=10 * T,
    fps=30,
    seconds_per_year=0.2,
    num_greeks=100,
    num_trojans=100,
    position_spread=0.1,
    save_animation=True,
)
