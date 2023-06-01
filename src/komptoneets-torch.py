import numpy as np
import torch
import plotext as plt
import sys
from typing import Tuple, Sequence

# n'' left =
#    (2n[0] - 5n[1] + 4n[2] - n[3]) / dx^3
#
# n'' right =
#    (2n[-1] - 5n[-2] + 4n[-3] - n[-4]) / dx^3
#
# n' left =
#     (-3n[0] + 4n[1] - n[2]) / (2 * dx)
#
# n' right =
#     (3n[-1] - 4n[-2] + n[-3]) / (2 * dx)


def hex_to_rgb(hex: str) -> Tuple[int, int, int]:
    hex = hex.lstrip("#")
    hlen = len(hex)
    return tuple(int(hex[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))


def Derivatives(ni, dx: float):
    """Calculate the first and second derivatives of a 1D array

    For the second derivative, enforce n'' = 0 at the boundaries
    """
    # find the second derivative of ni
    d2ni_dx2 = torch.zeros_like(ni)
    d2ni_dx2[1:-1] = (ni[2:] - 2 * ni[1:-1] + ni[:-2]) / dx**2
    # # enforce n'' = 0 at the boundaries

    # find ni[0] and ni[-1] from n'' = 0 condition
    ni_left = (5 / 2) * ni[1] - 2 * ni[2] + (1 / 2) * ni[3]
    ni_right = (5 / 2) * ni[-2] - 2 * ni[-3] + (1 / 2) * ni[-4]

    # find the first derivative of ni
    dni_dx_left = (-3 * ni_left + 4 * ni[1] - ni[2]) / (2 * dx)
    dni_dx_right = (3 * ni_right - 4 * ni[-2] + ni[-3]) / (2 * dx)

    dni_dx = torch.zeros_like(ni)
    dni_dx[1:-1] = (ni[2:] - ni[:-2]) / (2 * dx)
    dni_dx[0] = dni_dx_left
    dni_dx[-1] = dni_dx_right
    # dni_dx = np.concatenate(([dni_dx_left], dni_dx, [dni_dx_right]))
    return dni_dx, d2ni_dx2


# right hand side
def RHS(n0, n, dn_dx, d2n_dx2, x, T):
    return (
        T * d2n_dx2
        + (torch.exp(x) + 3 * T) * dn_dx
        + 2 * torch.exp(x) * n * (2 + dn_dx)
        + 4 * torch.exp(x) * n**2
        # + 4 * np.exp(x) * n
        # - n / 10
        # + n0 / 100
    )


def Iteration(n0, n, n1, x, T, dt):
    """
    1. ntilde(t+1) = n(t) + dt * RHS(n1, dn1/dx, d2n1/dx2, x, T, dx)
    2. ndash(t+1/2) = 0.5 * (ntilde(t+1) + n(t))
    """
    dx: float = x[1] - x[0]
    dn1_dx, d2n1_dx2 = Derivatives(n1, dx)
    ntilde: np.ndarray = n + dt * RHS(n0, n1, dn1_dx, d2n1_dx2, x, T)
    return 0.5 * (ntilde + n)


def Final(n0, n, ndash, e, T, dt):
    """
    n(t+1) = n(t) + dt * RHS(ndash(t+1/2), e, T, dx)
    """
    dx = e[1] - e[0]
    dndash_dx, d2ndash_dx2 = Derivatives(ndash, dx)
    return n + dt * RHS(n0, ndash, dndash_dx, d2ndash_dx2, e, T)


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


def progressbar(ax, value, minimum=0, maximum=100, label=None, color="white"):
    ax.bar([value], minimum=minimum, marker="sd", orientation="h", color=color)
    ax.xlim(minimum, maximum)
    ax.yticks([])
    ax.xlabel(label)


if __name__ == "__main__":
    import time

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("No GPU found")

    T0 = 1e-2
    emin = 1e-4
    emax = 10
    x_arr = torch.linspace(np.log(emin), np.log(emax), 100000, device=device)
    e_arr = torch.exp(x_arr)
    n_arr = torch.zeros_like(e_arr, device=device)
    # Delta function:
    n_arr[torch.argmin(torch.abs(e_arr - T0))] = 1
    # Planck distribution:
    # n_arr = 1 / (np.exp(e_arr / T0) - 1)
    T = 0.2
    dt = 0.0000002

    n0_arr = n_arr.clone()
    # n_arr *= 0

    norm = np.trapz((n0_arr * e_arr**2).cpu(), e_arr.cpu())

    xmin, xmax = np.log10(emin), np.log10(emax)
    ymin, ymax = -4, 1

    # if number of steps is passed:
    if len(sys.argv) > 1:
        nsteps = int(sys.argv[1])
    else:
        nsteps = 1000000

    for i in range(nsteps):
        now = time.time()
        ndash_arr = Iteration(n0_arr, n_arr, n_arr, x_arr, T, dt)
        ndash_arr = Iteration(n0_arr, n_arr, ndash_arr, x_arr, T, dt)
        n_arr = Final(n0_arr, n_arr, ndash_arr, x_arr, T, dt)
        n_arr = torch.abs(n_arr)
        duration = time.time() - now

        if i % 100 == 0:
            plt.clf()
            plt.cld()
            plt.clt()
            print(f"{duration:.2e}")
            plt.plotsize(100, 30)
            plt.subplots(2, 1)
            ax1 = plt.subplot(1, 1)
            ax1.theme("pro")
            ax1.plotsize(100, 20)
            # xs = np.logspace(-3, -1)
            # ys = 1e-3 * (xs / xs[0]) ** 3
            # ax1.plot(xs, ys)

            ax2 = plt.subplot(2, 1)
            ax2.theme("pro")
            ax2.plotsize(100, 5)
            progressbar(ax2, i * dt, 0, nsteps * dt, "time", "white")

            xs = e_arr[::100].cpu()
            ys = n_arr[::100].cpu()
            n_target = 1 / (np.exp(xs / T) - 1)
            n_target /= np.trapz(n_target * xs**2, xs)
            ax1.plot(xs, xs**3 * n_target + 1e-8, color="blue")

            ax1.plot(
                xs,
                xs**3 * ys / norm + 1e-8,
                color=hex_to_rgb("#d62728"),
            )
            ax1.xticks(*GoodTicks(emin, emax))
            ax1.yticks(*GoodTicks(10**ymin, 10**ymax))
            ax1.xlabel("e [me c^2]")
            ax1.ylabel("e^2 dn/de")
            ax1.xscale("log")
            ax1.yscale("log")
            ax1.ylim(ymin, ymax)
            ax1.xlim(xmin, xmax)
            plt.show()
