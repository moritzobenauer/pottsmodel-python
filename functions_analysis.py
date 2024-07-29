"""

MLO @ Princeton 2024
MC Simulation for Q-State Potts Model with Wang Landau Algorithm

Accounts for possible high exponents by calculating only ln values according to DOI: 10.1119/1.1707017
"""

import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt


def Normalize(array: pd.DataFrame, Q: int, size: int):
    """Normalize lng(E) data

    Args:
        array (pd.DataFrame): results from the WLA analysis
        Q (int): number of possible Q states
        size (int): lattice size

    Returns:
        _type_: returns x,y = E and lng(E)
    """
    p = array["lng(E)"][0]
    normalized = array["lng(E)"] - p + np.log(Q)
    return (array["E"], normalized)


def MirrorDataAndNormalize(array: pd.DataFrame):
    """Normalize lng(E) data and mirror it for Z2 symmetry of Q2 Potts Model

    Args:
        array (pd.DataFrame): results from the WLA analysis

    Returns:
        _type_: returns x,y = E and lng(E)
    """

    p = array["lng(E)"][0]
    normalized = array["lng(E)"] - p + np.log(2)
    x1 = np.array(array["E"])
    y1 = np.array(normalized)

    x2 = np.array(-array["E"])
    x2 = x2[:-1]
    x2 = np.flip(x2)
    z = np.array(array["lng(E)"])
    z = z[:-1]
    z = np.flip(z)

    p = z[-1]
    y2 = z - p + np.log(2)

    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    return (x, y)


def MLOThermo(T, energies, lnge, N, k: float = 1):
    """Calculate thermodynamic data

    Args:
        T (_type_): Temperature
        energies (_type_): E
        lnge (_type_): lng(E)
        N (_type_): number of lattice sites
        k (float, optional): Boltzmann constant. Defaults to 1.

    Returns:
        _type_: (F,U,C,S) thermodynamical data
    """
    # Find the maximum exponent lambda (DOI: 10.1119/1.1707017)

    energies = np.array(energies) * N**2

    exponents = []
    for L, E in zip(lnge, energies):
        exponents.append(L - E / (k * T))
    exponents = np.array(exponents)
    maxval = np.max(exponents)

    # Calculate F(T), U(T), C(T), and S(T)
    sigma = 0
    mu = 0
    kappa = 0
    for L, E in zip(lnge, energies):
        exponent = np.exp(L - E / (k * T) - maxval)
        sigma += exponent
        mu += exponent * E
        kappa += exponent * E * E

    lnZ = maxval + np.log(sigma)
    F = (-k * T * lnZ) / N

    U = mu / (sigma * N)

    C = ((kappa / sigma) - (mu / sigma) ** 2) / (k * T * N)

    S = (U - F) / T

    return (F, U, C, S)


def BoltzmannDist(energies, lnge, N, T, color, label):
    """Calculate Boltzmann Distributions

    Args:
        energies (_type_): E
        lnge (_type_): lng(E)
        N (_type_): lattice sites
        T (_type_): temperature
        color (_type_): color
        label (_type_): sample name or label
    """
    k = 1
    energies = np.array(energies) * N * N
    exponents = []
    for L, E in zip(lnge, energies):
        exponents.append(L - E / (k * T))
    exponents = np.array(exponents)
    maxval = np.max(exponents)

    # Calculate g(e)*exp(-E/T)
    dist = []
    for L, E in zip(lnge, energies):
        dist.append(np.exp(L - E / (k * T) - maxval))
    plt.plot(energies / N, dist, "ko-", color=color, label=label, alpha=0.6)
    plt.xlim(-2.1, 0)
