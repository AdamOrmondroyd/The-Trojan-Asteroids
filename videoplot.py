"""
Generates a video of the asteroids in the stationary frame.

Asteroids are perturbed about L4 and L5, and their behaviour in the orbital plane is recorded
as an mp4 file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from stationaryframe import StationaryAsteroid
import multiprocessing

ast = StationaryAsteroid()

# Run for 10 planetary orbits
run_time = 10 * ast.T
fps = 30
seconds_per_year = 0.2
num_greeks = 100
num_trojans = 100
position_spread = 0.1
velocity_spread = 0.1
save_animation = True
file_name = "outputs\\movie.mp4"

num_points = int(run_time * fps * seconds_per_year)
ts = np.linspace(0, run_time, num_points)

greek_xs = np.zeros((num_greeks, num_points))
greek_ys = np.zeros((num_greeks, num_points))
greek_zs = np.zeros((num_greeks, num_points))

greek_input = np.full(num_greeks, True, dtype=bool)

trojan_xs = np.zeros((num_trojans, num_points))
trojan_ys = np.zeros((num_trojans, num_points))
trojan_zs = np.zeros((num_trojans, num_points))

trojan_input = np.full(num_trojans, False, dtype=bool)


def random_asteroid_wrapper(greek):
    """Wrapper returning the solution of an asteroid perturbed about starting_point"""
    r_offset = (np.random.rand(3) - 0.5) * position_spread
    v_offset = (np.random.rand(3) - 0.5) * velocity_spread

    if greek:
        r_0 = ast.l4(0) + r_offset
    else:
        r_0 = ast.l5(0) + r_offset

    return ast.trajectory(ts, r_0, v_0=ast.omega_cross(r_0) + v_offset)


if __name__ == "__main__":
    pool = multiprocessing.Pool()

    greek_sols = pool.map(random_asteroid_wrapper, greek_input)

    trojan_sols = pool.map(random_asteroid_wrapper, trojan_input)
    pool.close()

    for i in range(num_greeks):
        greek_xs[i] = greek_sols[i].y[0]
        greek_ys[i] = greek_sols[i].y[1]
        greek_zs[i] = greek_sols[i].y[2]

    for i in range(num_trojans):
        trojan_xs[i] = trojan_sols[i].y[0]
        trojan_ys[i] = trojan_sols[i].y[1]
        trojan_zs[i] = trojan_sols[i].y[2]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6), aspect="equal")

    (sun_line,) = ax.plot(
        [], [], label="sun", color="orange", marker="*", markersize=20, linestyle="None"
    )
    (p_line,) = ax.plot(
        [],
        [],
        label="Planet",
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

    ax.set(title="Stationary frame", xlabel="x/au", ylabel="y/au")

    ax.legend(loc="upper right", frameon=False, prop={"size": 10})

    def animate(i):
        sun_position = ast.r_sun(ts[i])
        sun_line.set_data(sun_position[0], sun_position[1])
        p_position = ast.r_p(ts[i])
        p_line.set_data(p_position[0], p_position[1])

        l4 = ast.l4(ts[i])
        l4_line.set_data(l4[0], l4[1])
        l5 = ast.l5(ts[i])
        l5_line.set_data(l5[0], l5[1])

        greeks_line.set_data(greek_xs[:, i], greek_ys[:, i])
        trojans_line.set_data(trojan_xs[:, i], trojan_ys[:, i])

        time_text.set_text(str(np.round(ts[i], 1)) + " years")
        return (
            sun_line,
            p_line,
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

        anim.save(file_name, writer=writer)
        print("saved")
    plt.show()
