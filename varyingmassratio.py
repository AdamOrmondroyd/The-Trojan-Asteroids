import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import time
import multiprocessing
from scipy.optimize import curve_fit

fig, ax = plt.subplots(1, 3, figsize=(12, 5))


def wander_wrapper(m):
    ast = RotatingAsteroid(M_P=m)
    print(ast.M_P)
    end_time = 100 * ast.T
    points_per_year = 100
    ts = np.linspace(0, end_time, int(end_time * points_per_year))
    return ast.wander(
        ts,
        r_0=ast.L4 * (1 + 0.0001 / np.linalg.norm(ast.L4)),
        v_0=np.array([0, 0, 0]),
        stability_point=ast.L4,
    )


def quadratic(x, a, b, c):
    """quadratic for curve fitting"""
    # return a * x ** 2 + b * x + c
    # return np.cosh((x - a) / b) + c
    # return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    return a * (x - b) ** 6 + c


### Smaller masses ###

if __name__ == "__main__":
    m_min = 0.00001
    m_max = 0.002
    points = 30
    ms = np.linspace(m_min, m_max, points)

    tic = time.time()

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, ms)
    pool.close()

    ax[0].plot(ms, wanders, label="wanders", marker="+", color="c", linestyle="None")

    ax[0].set(
        title="M$_\mathrm{{P}}$ from {}M$_\odot$ to {}M$_\odot$".format(m_min, m_max),
        xlabel="M$_{\mathrm{P}}$/M$_{\odot}}$",
        ylabel="Maximum wander / au",
    )

    ### Medium masses ###

    m_min = 0.001
    m_max = 0.035
    ms = np.linspace(m_min, m_max, points)

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, ms)
    pool.close()

    (a, b, c), pcov = curve_fit(quadratic, ms, wanders)

    print(a)
    print(b)
    print(c)

    equation_string = "{:.1f}m$^2$ {:+.1f}m {:+.3e}".format(a, b, c)

    print("Minimum at {} ± {}".format(a, np.sqrt(pcov[0, 0])))

    ax[1].plot(
        ms, wanders, label="wanders", marker="+", color="c", linestyle="None",
    )
    ax[1].plot(
        ms, quadratic(ms, a, b, c), label=equation_string, color="k", linestyle="--",
    )

    ax[1].set(
        title="M$_\mathrm{{P}}$ from {}M$_\odot$ to {}M$_\odot$".format(m_min, m_max),
        xlabel="M$_{\mathrm{P}}$/M$_{\odot}}$",
        ylabel="Maximum wander / au",
        xticks=np.linspace(0, m_max, 8),
    )

    ### Big masses ###
    m_min = 0.001
    m_max = 0.042
    ms = np.linspace(m_min, m_max, points)

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, ms)
    pool.close()

    toc = time.time()
    print("Time taken: {:.1f} s".format(toc - tic))

    ax[2].plot(ms, wanders, label="wanders", marker="+", color="c", linestyle="None")

    ax[2].set(
        title="M$_\mathrm{{P}}$ from {}M$_\odot$ to {}M$_\odot$".format(m_min, m_max),
        xlabel="M$_{\mathrm{P}}$/M$_{\odot}}$",
        ylabel="Maximum wander / au",
    )

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=len(labels),
    )

    fig.tight_layout()

    fig.subplots_adjust(bottom=0.2)

    filename = "plots\\mass_wanders"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")
    plt.show()
