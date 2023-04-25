import sys
import os
import plotext as plt
import numpy as np
from typing import Tuple, Sequence


def GoodTicks(xmin: float, xmax: float) -> Tuple[Sequence[float], Sequence[float]]:
    ticks = [
        10**i
        for i in range(int(np.ceil(np.log10(xmin))), int(np.floor(np.log10(xmax))) + 1)
    ]
    ticklabels = [
        f"1e{int(np.log10(t))}"
        if (t < 0.01 or t > 1000)
        else (int(t) if t >= 1 else float(t))
        for t in ticks
    ]
    return ticks, ticklabels


def hex_to_rgb(hex: str) -> Tuple[int, int, int]:
    hex = hex.lstrip("#")
    hlen = len(hex)
    return tuple(int(hex[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
    else:
        raise ValueError("No filename given")

    fname_prev = None
    while True:
        # if directory exists
        if not os.path.isdir(fpath):
            fname = None
            print("No directory")
        else:
            files = [f for f in os.listdir(fpath) if f.startswith("time_")]
            if len(files) > 1:
                sorted_files = sorted(
                    files, key=lambda x: int(x.replace(".csv", "").split("_")[1])
                )
                fname = sorted_files[-2]
                if fname == fname_prev:
                    continue
                else:
                    fname_prev = fname
                e_arr, n_arr = np.loadtxt(
                    f"{fpath}/{fname}",
                    unpack=True,
                    delimiter=",",
                    dtype=float,
                    comments="#",
                )
                with open(f"{fpath}/{fname}", "r") as f:
                    metadata = {}
                    for line in f:
                        if line.startswith("##"):
                            key, value = line[2:].strip().split(":")
                            metadata[key] = float(value)
                        else:
                            break

                emin = e_arr[0]
                emax = e_arr[-1]
                Nx = len(e_arr)

                norm = np.trapz(n_arr * e_arr**2, e_arr)

                xmin, xmax = np.log10(emin), np.log10(emax)
                ymin, ymax = -4, 1
                y = e_arr**3 * n_arr / norm
                y_target = e_arr**2 / (np.exp(e_arr / metadata["temperature"]) - 1)
                y_target *= e_arr / np.trapz(y_target, e_arr)

                y = np.where(y > 10**ymin, y, 1e-5)
                y_target = np.where(y_target > 10**ymin, y_target, 1e-5)

                plt.plotsize(100, 30)
                plt.theme("pro")

                plt.plot(e_arr, y, color=hex_to_rgb("#d62728"))
                plt.plot(e_arr, y_target, color="blue")
                plt.xticks(*GoodTicks(emin, emax))
                plt.yticks(*GoodTicks(10**ymin, 10**ymax))
                plt.xlabel("e [me c^2]")
                plt.ylabel("e^2 dn/de")
                plt.xscale("log")
                plt.yscale("log")
                plt.ylim(ymin, ymax)
                plt.xlim(xmin, xmax)
                plt.title(fname + f", t = {metadata['time']:.2f}")
                plt.show()
            else:
                fname = None
                print("No files yet")
        plt.clf()
        plt.cld()
        plt.clt()
        plt.sleep(0.1)
