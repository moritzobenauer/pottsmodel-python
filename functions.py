"""

MLO @ Princeton 2024
MC Simulation for Q-State Potts Model with Wang Landau Algorithm

v2.0.1  object orienetd approach

"""

import random
import numpy as np  # type: ignore
from numpy.random import rand  # type: ignore
import pandas as pd  # type: ignore
from numba import jit  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

"""
Classes
"""


class Lattice:
    def __init__(self, size: int = 10):
        self.size = size
        self.particles = size**2
        self.grid = np.zeros((size, size))

    def Randomize(self):
        with np.nditer(self.grid, op_flags=["readwrite"]) as it:
            for x in it:
                x[...] = random.choice([-1, 1])

    def GridEnergy(self, J):
        return CalculateEnergy(self.grid, J)

    def NNEnergy(self, J, x, y):
        s = self.grid[x, y]
        nnl, nns = GetNNL(self.grid, x, y)
        energy_contributions = np.array(
            [-J if s == neighbor else 0 for neighbor in nns]
        )  # Kronecker Delta Function
        E = np.sum(energy_contributions)
        return E

    def RandPos(self):
        """
        Random lattice point selection and returns random position of the lattice
        """
        i, j = np.random.randint(0, self.size, size=2)
        return (i, j)

    def FlipRandPos(self, q):
        i, j = np.random.randint(0, self.size, size=2)
        old_state = self.grid[i, j]
        possible_states = np.arange(0, q, 1)
        new_state = np.random.choice(possible_states)
        return (i, j, old_state, new_state)

    def SetPosition(self, i, j, k):
        self.grid[i, j] = k

    def GetMag(self):
        """
        Returns the overall magnezization M of the lattice. Probably only useful for Q=2 (Ising)
        """
        M = ArraySum(self.grid) / self.particles
        return M


"""
Functions
"""


@jit(nopython=True)
def ArraySum(lattice: np.array):
    return np.sum(lattice)


@jit(nopython=True)
def GetNNL(lattice: np.array, i: int, j: int):
    """Returns nearest neighbors (version 1.1 with easier boundary conditions)
    Args:
        lattice (np.array): lattice/grid
        i (int): i dimension
        j (int): j dimension
        i=j for all cases but this is a more genreal version for non-square lattices

    Returns:
        nnl: nearest neighbor list
        nns: nearest neighbor states
    """
    N = np.shape(lattice)[0]
    nnl = [[(i + 1) % N, j], [(i - 1) % N, j], [i, (j + 1) % N], [i, (j - 1) % N]]

    nns = [lattice[index[0], index[1]] for index in nnl]
    return nnl, np.array(nns)


@jit(nopython=True)
def CalculateEnergy(lattice: np.array, J: float = 1.0) -> float:
    """Calculate energy of a given system without external field contributions

    Args:
        lattice (np.array): lattice/grid
        J (float): coupling constant. Defaults to 1.

    Returns:
        float: lattice energy
    """
    energy = 0.0
    for x in range(lattice.shape[0]):
        for y in range(lattice.shape[1]):
            s = lattice[x, y]  # Current lattice position
            nnl, nns = GetNNL(lattice, x, y)  # Get current lattice position's neighbors
            energy_contributions = np.array(
                [-J if s == neighbor else 0 for neighbor in nns]
            )  # Kronecker Delta Function
            energy += np.sum(energy_contributions)  # Sum over all N lattice sites
    return energy / 2.0


@jit(nopython=True)
def GetDeltaIndex(array: np.array, number: float) -> int:
    """Returns the belonging of an energy value (number) to a energy bin (array)


    Args:
        array (np.array): energy bins
        number (float): current energy

    Returns:
        int: index of the current energy
    """
    index = np.searchsorted(array, number, side="right")
    return index - 1


def PrintLNF(lnf: float):
    """Just a pretty print function

    Args:
        lnf (float): Current ln(f) value
    """
    print("-------------------")
    print("ln(f): ", lnf)
    print("-------------------")


"""
def BinChecking(MAX_STEPS, lattice, ref, q, LB, UB):
    print("-------------------")
    print("Running Bin Checker")
    N = len(ref)
    lnge = np.zeros(N)
    hist = np.zeros(N)
    lnf = 1.0

    for iter in range(MAX_STEPS):

        ene = lattice.GridEnergy(J=1)

        EnergyCheck = False
        while not EnergyCheck:
            i, j, old_state, new_state = lattice.FlipRandPos(q)
            lattice.SetPosition(i, j, new_state)
            enew = lattice.GridEnergy(1)
            if enew <= UB and enew >= LB:
                EnergyCheck = True
            else:
                lattice.SetPosition(i, j, old_state)

        index_eold = GetDeltaIndex(ref, ene)
        index_enew = GetDeltaIndex(ref, enew)

        dos_ratio = np.exp(lnge[index_eold] - lnge[index_enew])  # Difference in DOS

        if dos_ratio >= 1.0 or np.random.rand() < dos_ratio:  # WLA Criterion
            hist[index_enew] += 1
            lnge[index_enew] += lnf
            # print(enew)

        else:  # Update the current bins and change back to the original k configuration
            hist[index_eold] += 1
            lnge[index_eold] += lnf
            lattice.SetPosition(i, j, old_state)

    empty_bins = [i for i, e in enumerate(hist) if e == 0]
    empty_energies = [ref[e] for e in empty_bins]
    print("The following bins will be excluded:")
    print(empty_bins)
    print("This corresponds to the following energies:")
    print(empty_energies)
    print("-------------------")

    return (empty_bins, empty_energies)
"""


def WangLandau(
    lattice: Lattice,
    ref: np.array,
    MAX_STEPS: int = 10e8,
    NBINS: int = 500,
    INTERVAL: int = 1000,
    L: int = 10,
    FLATNESS: float = 0.8,
    CONTROLF: float = 10e-8,
    q: int = 2,
    LB: float = -2.0,
    UB: float = 0.0,
):
    """The actual Wang Landau Algorithm

    Args:
        lattice (Lattice class): the grid/lattice object
        ref (np.array): energy array (probably from -2 to 0)
        MAX_STEPS (int, optional): Maximum steps for convergence for every lnf step. Defaults to 10e8.
        NBINS (int, optional): Number of energy bins. Defaults to 500.
        INTERVAL (int, optional): Printing Interval for Updates. Defaults to 1000.
        L (int, optional): Lattice Size. Defaults to 10.
        FLATNESS (float, optional):  WLA flatness. Defaults to 0.8.
        CONTROLF (float, optional): WLA final lnf=10e-8 criterion.. Defaults to 10e-8.
        DIRECTORY_NAME (str, optional): Current Experiment name. Defaults to "samplerun".
        q (int, optional): Number of possible states. Defaults to 8.
        LB (float, optional): Lower energy bound. Defaults to -2.0.
        UB (float, optional): Upper energy bound. Defaults to 0.0.

    Returns:
        energy bins, lnge, last histogram
    """

    MCS = L**2
    N = NBINS
    lnge = np.zeros(NBINS)  # Initial DOS = 0
    lnf = 1.0  # Initial f = e
    MAX_STEPS = int(MAX_STEPS)

    possible_states = np.arange(
        0, q, 1
    )  # There are Q possible states in the Q-state Potts Model
    print(f"Possible States (Q={q}):")
    print(possible_states)

    LB_INDEX = GetDeltaIndex(ref, LB)  # lower energy bound
    UB_INDEX = GetDeltaIndex(ref, UB)  # upper energy bound

    print("Lower Energy Bound:", LB)
    print("Lower Energy Index: ", LB_INDEX)

    print("Upper Energy Bound:", UB)
    print("Upper Energy Index: ", UB_INDEX)

    """
    Checking the integrity of all available bins
    Also exclude upper and lower energy boundaries
    """
    # empty_bins, empty_energies = BinChecking(MAX_STEPS, lattice, ref, q, LB, UB)
    mask = np.ones(len(ref), dtype=bool)
    exclude_bins = [i for i, e in enumerate(ref) if e > UB or e < LB]
    mask[exclude_bins] = False

    print("Maximal ln(f)", CONTROLF)

    while lnf > CONTROLF:  # This loops controls the precision of the algorithm
        PrintLNF(lnf)
        hist = np.zeros(NBINS)  # Resetting the histogram

        for iter in range(
            MAX_STEPS
        ):  # Abort if no convergence is reached after MAX_STEPS

            ene = lattice.GridEnergy(J=1)

            EnergyCheck = False
            while not EnergyCheck:
                i, j, old_state, new_state = lattice.FlipRandPos(q)
                lattice.SetPosition(i, j, new_state)
                enew = lattice.GridEnergy(1)
                if enew <= UB and enew >= LB:
                    EnergyCheck = True
                else:
                    lattice.SetPosition(i, j, old_state)

            index_eold = GetDeltaIndex(ref, ene)
            index_enew = GetDeltaIndex(ref, enew)

            dos_ratio = np.exp(lnge[index_eold] - lnge[index_enew])  # Difference in DOS

            if dos_ratio >= 1.0 or np.random.rand() < dos_ratio:  # WLA Criterion
                hist[index_enew] += 1
                lnge[index_enew] += lnf
                # print(enew)

            else:  # Update the current bins and change back to the original k configuration
                hist[index_eold] += 1
                lnge[index_eold] += lnf
                lattice.SetPosition(i, j, old_state)

            if iter % MCS == 0:
                actual_hist = hist[mask]
                if (
                    np.min(actual_hist) > np.sum(actual_hist) / NBINS * FLATNESS
                ):  # WLA FLATNESS Criterion
                    print("Reached convergence after", iter, "steps.")
                    print(
                        "Hist FLATNESS: ",
                        np.round(
                            np.min(actual_hist)
                            * NBINS
                            / (np.sum(actual_hist) * FLATNESS),
                            3,
                        ),
                    )
                    lnf /= 2  # f(t+1) = sqrt(f(t))
                    break  # Escape the loop and start with new lnf

            if (
                iter % (MCS * INTERVAL) == 0
            ):  # Printing current progress every MCS*INTERVAL steps
                actual_hist = hist[mask]
                print("Current Iteration: ", iter)
                print(
                    "Hist FLATNESS: ",
                    np.round(
                        np.min(actual_hist) * NBINS / (np.sum(actual_hist) * FLATNESS),
                        3,
                    ),
                )
                print("Smallest Bin: ", np.argmin(actual_hist))
                print("Current Energy: ", enew)

            if (
                iter == MAX_STEPS - 1
            ):  # If no convergence is reached, stop the sampling by setting lnf=0 --> breaks out of the while loop
                if lnf == 1.0:
                    empty_bins = [i for i, e in enumerate(hist) if e == 0]
                    print("Excluded the following bins for subsequent runs:")
                    print(empty_bins)
                    mask[empty_bins] = False
                    lnf /= 2

                else:
                    print("Reached no convergence after", MAX_STEPS, "steps.")
                    print("Reached lnf=", lnf)
                    print("Smallest bin:", np.argmin(actual_hist))
                    print("with count:", np.min(actual_hist))
                    lnf = 0

        actual_hist = hist[mask] / np.max(hist[mask])
        actual_lnge = lnge[mask]
        actual_ref = ref[mask] / lattice.particles

    return (
        actual_ref,
        actual_lnge,
        actual_hist,
    )  # Returns the energy bins ref/N, the DOS lnge, and the last normalized histogram hist
