import numpy as np
import matplotlib.pyplot as plt
from rotatingframe import RotatingAsteroid
import time
import multiprocessing
from scipy.optimize import curve_fit

fig, ax = plt.subplots(1, 2, figsize=(12, 5))


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


def exponential_decay(x, a, b):
    """exponential decay for curve fitting"""
    return a / x ** 0.5 + b


### Smaller masses ###

if __name__ == "__main__":
    m_min = 0.00001
    m_max = 0.001
    points = 100
    ms = np.linspace(m_min, m_max, points)

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, ms)
    pool.close()

    (a, b), pcov = curve_fit(exponential_decay, ms, wanders)

    equation_string = "{:.2e}/√m + {:.2e}".format(a, b)

    ax[0].plot(ms, wanders, label="wanders", color="c", marker="+", linestyle="None")
    ax[0].plot(
        ms,
        exponential_decay(ms, a, b),
        label=equation_string,
        color="k",
        linestyle="--",
    )

    ax[0].set(
        title="M$_\mathrm{{P}}$ from {}M$_\odot$ to {}M$_\odot$".format(m_min, m_max),
        xlabel="M$_{\mathrm{P}}$/M$_{\odot}}$",
        ylabel="wander / au",
    )

    ### Larger masses ###

    m_min = 0.0
    m_max = 0.044
    ms = np.linspace(m_min, m_max, points)

    pool = multiprocessing.Pool()
    wanders = pool.map(wander_wrapper, ms)
    pool.close()

    ax[1].plot(
        ms, wanders, label="wanders", marker="+", color="c", linestyle="None",
    )

    ax[1].set(
        title="M$_\mathrm{{P}}$ from {}M$_\odot$ to {}M$_\odot$".format(m_min, m_max),
        xlabel="M$_{\mathrm{P}}$/M$_{\odot}}$",
        ylabel="wander / au",
        # xticks=np.linspace(0, m_max, 8),
    )

    handles, labels = ax[0].get_legend_handles_labels()
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
