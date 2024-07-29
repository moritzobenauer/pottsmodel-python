import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from functions_analysis import *

if __name__ == "__main__":
    data10 = pd.read_csv("examples/10x10_Q8/out_final.txt")
    data16 = pd.read_csv("examples/16x16_Q8/out_final.txt")

    data = [data10, data16]
    colors = ["orange", "blue"]
    gridsizes = [10, 16]
    latticesize = [x**2 for x in gridsizes]

    tc = 1.0 / (np.log(1 + np.sqrt(8)))

    for color, N, g, d in zip(colors, latticesize, gridsizes, data):
        x, y = Normalize(d, 8, N)
        plt.plot(
            x * N,
            y,
            "bo-",
            label=f"{g}x{g}, Flatness: 0.8",
            alpha=0.6,
            color=color,
            markersize=4,
        )

    plt.ylabel("ln g(E)")
    plt.xlabel("E")
    plt.legend(loc="best")
    plt.savefig("figures/q8_lnge.png")
    plt.clf()

    temps = np.linspace(0.1, 2.0, 500)  # T varies from 0.4 to 8

    for color, N, g, d in zip(colors, latticesize, gridsizes, data):
        x, y = Normalize(d, 8, N)
        plots = []
        for T in temps:
            plots.append(MLOThermo(T, x, y, N))
        plots = np.array(plots)
        fig = plt.figure(1)
        plt.plot(
            temps,
            plots[:, 2],
            "ko--",
            label=f"{g}x{g}, Flatness: 0.8, HI",
            alpha=0.4,
            color=color,
            markersize=4,
        )
        plt.axvline(tc, color="k")
        plt.xlabel("T")
        plt.ylabel("C(T)/N")
        plt.legend(loc="best")
    plt.savefig("figures/q8_C.png")
    plt.clf()

    for color, N, g, d in zip(colors, latticesize, gridsizes, data):
        x, y = Normalize(d, 8, N)
        plots = []
        for T in temps:
            plots.append(MLOThermo(T, x, y, N))
        plots = np.array(plots)
        fig = plt.figure(1)
        plt.plot(
            temps,
            plots[:, 0],
            "ko--",
            label=f"{g}x{g}, Flatness: 0.8, HI",
            alpha=0.4,
            color=color,
            markersize=4,
        )
        plt.axvline(tc, color="k")
        plt.xlabel("T")
        plt.ylabel("F(T)/N")
        plt.legend(loc="best")
    plt.savefig("figures/q8_F.png")
    plt.clf()

    for color, N, g, d in zip(colors, latticesize, gridsizes, data):
        x, y = Normalize(d, 8, N)
        plots = []
        for T in temps:
            plots.append(MLOThermo(T, x, y, N))
        plots = np.array(plots)
        fig = plt.figure(1)
        plt.plot(
            temps,
            plots[:, 1],
            "ko--",
            label=f"{g}x{g}, Flatness: 0.8, HI",
            alpha=0.4,
            color=color,
            markersize=4,
        )
        plt.axvline(tc, color="k")
        plt.xlabel("T")
        plt.ylabel("U(T)/N")
        plt.legend(loc="best")
    plt.savefig("figures/q8_U.png")
    plt.clf()

    for color, N, g, d in zip(colors, latticesize, gridsizes, data):
        x, y = Normalize(d, 8, N)
        plots = []
        for T in temps:
            plots.append(MLOThermo(T, x, y, N))
        plots = np.array(plots)
        fig = plt.figure(1)
        plt.plot(
            temps,
            plots[:, 3],
            "ko--",
            label=f"{g}x{g}, Flatness: 0.8, HI",
            alpha=0.4,
            color=color,
            markersize=4,
        )
        plt.axvline(tc, color="k")
        plt.xlabel("T")
        plt.ylabel("S(T)/N")
        plt.legend(loc="best")
    plt.savefig("figures/q8_S.png")
    plt.clf()
