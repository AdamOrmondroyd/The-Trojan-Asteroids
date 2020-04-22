import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from stationaryframe import asteroid, r_sun, r_j, l_4, l_5, omega_cross
from constants import L4, L5, R, R_SUN, R_J, T, W
import multiprocessing

run_time = 10 * T
fps = 30
seconds_per_year = 0.2
num_greeks = 5
num_trojans = 5
position_spread = 0.1
velocity_spread = 0.1
save_animation = True
file_name = "movie.mp4"

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
        r_0 = L4 + r_offset
    else:
        r_0 = L5 + r_offset

    v_0 = omega_cross(r_0) + v_offset

    return asteroid(ts, r_0, v_0)


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
    # ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6), aspect="equal")
    ax = fig.add_subplot(111, projection="3d")

    sun_line = ax.scatter([], [], label="sun", color="orange", marker="*")
    j_line = ax.scatter([], [], label="Jupiter", color="red", marker="o")
    l4_line = ax.scatter([], [], label="L$_4$", color="blue", marker="+")
    l5_line = ax.scatter([], [], label="L$_5$", color="red", marker="+")
    greeks_line = ax.scatter([], [], label="Greeks", color="green", marker="o")
    trojans_line = ax.scatter([], [], label="Trojans", color="magenta", marker="o",)

    # time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.set(title="Stationary frame", xlabel="x/AU", ylabel="y/AU", zlabel="z/AU")

    # ax.legend(loc="upper right", frameon=False, prop={"size": 10})

    def animate(i):
        sun_position = r_sun(ts[i])
        sun_line.offsets3d = (sun_position[0], sun_position[1], sun_position[2])
        j_position = r_j(ts[i])
        j_line.offsets3d = (j_position[0], j_position[1], j_position[2])

        l4 = l_4(ts[i])
        l4_line.offsets3d = (l4[0], l4[1], l4[2])
        l5 = l_5(ts[i])
        l5_line.offsets3d = (l5[0], l5[1], l5[2])

        greeks_line.offsets3d = (greek_xs[:, i], greek_ys[:, i], greek_zs[:, i])
        trojans_line.offsets3d = (trojan_xs[:, i], trojan_ys[:, i], trojan_zs[:, i])

        # time_text.set_text(str(np.round(ts[i], 1)) + " years")
        # return (
        #     sun_line,
        #     j_line,
        #     l4_line,
        #     l5_line,
        #     greeks_line,
        #     trojans_line,
        #     # time_text,
        # )

    anim = FuncAnimation(
        fig,
        animate,
        frames=int(num_points),  # supplies range(frames) to animate
        interval=1 / fps,  # time between frames
        blit=False,
    )
    print("animated")

    if save_animation:
        writer = animation.FFMpegWriter(
            fps=fps, metadata=dict(artist="Adam Ormondroyd"),  # bitrate=1800
        )

        anim.save(file_name, writer=writer)
        print("saved")
    plt.show()
