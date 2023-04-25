import numpy as np
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


def Derivatives(ni: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the first and second derivatives of a 1D array

    For the second derivative, enforce n'' = 0 at the boundaries

    Parameters
    ----------
    ni : np.ndarray
        1D array of values to take the derivatives of
    dx : float
        grid spacing

    Returns
    -------
    dni_dx : np.ndarray
        1D array of first derivatives
    d2ni_dx2 : np.ndarray
        1D array of second derivatives
    """
    # find the second derivative of ni
    d2ni_dx2 = (ni[2:] - 2 * ni[1:-1] + ni[:-2]) / dx**2
    # enforce n'' = 0 at the boundaries
    d2ni_dx2 = np.concatenate(([0], d2ni_dx2, [0]))

    # find ni[0] and ni[-1] from n'' = 0 condition
    ni_left = (5 / 2) * ni[1] - 2 * ni[2] + (1 / 2) * ni[3]
    ni_right = (5 / 2) * ni[-2] - 2 * ni[-3] + (1 / 2) * ni[-4]

    # find the first derivative of ni
    dni_dx_left = (-3 * ni_left + 4 * ni[1] - ni[2]) / (2 * dx)
    dni_dx_right = (3 * ni_right - 4 * ni[-2] + ni[-3]) / (2 * dx)
    dni_dx = (ni[2:] - ni[:-2]) / (2 * dx)
    dni_dx = np.concatenate(([dni_dx_left], dni_dx, [dni_dx_right]))
    return dni_dx, d2ni_dx2


# right hand side
def RHS(
    n0: np.ndarray,
    n: np.ndarray,
    dn_dx: np.ndarray,
    d2n_dx2: np.ndarray,
    x: np.ndarray,
    T: float,
) -> np.ndarray:
    return (
        T * d2n_dx2
        + (np.exp(x) + 3 * T) * dn_dx
        + 2 * np.exp(x) * n * (2 + dn_dx)
        + 4 * np.exp(x) * n**2
        # + 4 * np.exp(x) * n
        # - n / 10
        # + n0 / 100
    )


def CN_predictor(
    n0: np.ndarray, n: np.ndarray, n1: np.ndarray, x: np.ndarray, T: float, dt: float
) -> np.ndarray:
    """
    Use Crank-Nicolson method to predict the next value of a given array of values.

    Parameters
    ----------
    n0 : 1D numpy.ndarray
        Function n at time 0
    n : 1D numpy.ndarray
        Function n at time t
    n1 : 1D numpy.ndarray
        Function n either at time t (first iteration) or t + 1/2 (subsequent iterations)
    x : 1D numpy.ndarray
        The discretized grid (energy).
    T : float
        The temperature.
    dx : float
        The step size for the grid.
    dt : float
        The time step.

    Returns
    -------
    numpy.ndarray
        An array of predicted values for n at time t + 1/2.

    Notes
    -----
    This function has two steps:
    1. ntilde(t+1) = n(t) + dt * RHS(n1, dn1/dx, d2n1/dx2, x, T, dx)
    2. ndash(t+1/2) = 0.5 * (ntilde(t+1) + n(t))
    """
    dx: float = x[1] - x[0]
    dn1_dx, d2n1_dx2 = Derivatives(n1, dx)
    ntilde: np.ndarray = n + dt * RHS(n0, n1, dn1_dx, d2n1_dx2, x, T)
    return 0.5 * (ntilde + n)


def CN_corrector(
    n0: np.ndarray, n: np.ndarray, ndash: np.ndarray, e: np.ndarray, T: float, dt: float
) -> np.ndarray:
    """
    Use Crank-Nicolson method to correct the predicted value of a given array of values.

    Parameters
    ----------
    n0 : 1D numpy.ndarray
        Function n at time 0
    n : 1D numpy.ndarray
        Function n at time t.
    ndash : 1D numpy.ndarray
        The predicted value of function n at time t + 1/2.
    e : 1D numpy.ndarray
        The discretized grid (energy).
    T : float
        The temperature.
    dt : float
        The time step.

    Returns
    -------
    numpy.ndarray
        An array of corrected values for n at time t + 1.

    Notes
    -----
    This function calculates the value of n at time t + 1:
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
    T0 = 1e-2
    emin = 1e-4
    emax = 10
    x_arr = np.linspace(np.log(emin), np.log(emax), 1000)
    e_arr = np.exp(x_arr)
    n_arr = np.zeros_like(e_arr)
    # Delta function:
    n_arr[np.argmin(np.abs(e_arr - T0))] = 1
    # Planck distribution:
    # n_arr = 1 / (np.exp(e_arr / T0) - 1)
    T = 0.2
    dt = 0.0002

    n0_arr = n_arr.copy()
    # n_arr *= 0

    norm = np.trapz(n0_arr * e_arr**2, e_arr)

    xmin, xmax = np.log10(emin), np.log10(emax)
    ymin, ymax = -4, 1

    # if number of steps is passed:
    if len(sys.argv) > 1:
        nsteps = int(sys.argv[1])
    else:
        nsteps = 1000000

    for i in range(nsteps):
        ndash_arr = CN_predictor(n0_arr, n_arr, n_arr, x_arr, T, dt)
        ndash_arr = CN_predictor(n0_arr, n_arr, ndash_arr, x_arr, T, dt)
        ndash_arr = CN_predictor(n0_arr, n_arr, ndash_arr, x_arr, T, dt)
        n_arr = CN_corrector(n0_arr, n_arr, ndash_arr, x_arr, T, dt)
        n_arr = np.abs(n_arr)
        if np.any(np.isnan(n_arr)):
            raise ValueError("NaN encountered")
        elif np.any(n_arr < 0):
            raise ValueError("Negative value encountered", np.min(n_arr))

        if i % 100 == 0:
            plt.clf()
            plt.cld()
            plt.clt()
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

            n_target = 1 / (np.exp(e_arr / T) - 1)
            n_target /= np.trapz(n_target * e_arr**2, e_arr)
            ax1.plot(e_arr, e_arr**3 * n_target + 1e-8, color="blue")

            ax1.plot(
                e_arr, e_arr**3 * n_arr / norm + 1e-8, color=hex_to_rgb("#d62728")
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
