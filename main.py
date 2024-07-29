"""

MLO @ Princeton 2024
MC Simulation for Q-State Potts Model with Wang Landau Algorithm

v2.0.1  object orienetd approach

"""

import argparse
import os
from functions import *


def main():

    parser = argparse.ArgumentParser(description="WLA-2DIsingModel2024-MLO")
    parser.add_argument("-g", "--gridsize", type=int, help="gridsize", default=10)
    parser.add_argument(
        "-f", "--directoryname", type=str, help="directory name", default="WLA-RUN"
    )
    parser.add_argument("-z", "--flatness", type=float, help="flatness", default=0.8)
    parser.add_argument(
        "-m", "--finallnf", type=float, help="final lnf(f)", default=0.000001
    )
    parser.add_argument("-n", "--bins", type=int, help="number of bins", default=100)
    parser.add_argument(
        "-q", "--qstates", type=int, help="number of q states", default=2
    )

    args = parser.parse_args()

    DIRECTORY_NAME = args.directoryname
    L = args.gridsize
    N = args.bins
    FLATNESS = args.flatness
    Q = args.qstates
    FINAL_LNF = args.finallnf

    try:
        os.mkdir(DIRECTORY_NAME)
    except FileExistsError:
        pass

    maxsteps = 1e6
    LB = -2.0 * L**2
    UB = -0.0 * L**2

    ref = np.linspace(LB, UB, N)  # Setting up energy bins
    print("Number of Bins:", N)

    EnergyCheck = False
    while not EnergyCheck:
        x = Lattice(L)
        x.Randomize()
        initial_energy = x.GridEnergy(1)
        if initial_energy <= UB and initial_energy >= LB:
            EnergyCheck = True

    print("Found initial lattice. Energy: ", initial_energy)

    (REF, LNGE_A, HIST_A) = WangLandau(
        x,
        ref,
        maxsteps,
        NBINS=N,
        INTERVAL=100,
        L=L,
        FLATNESS=FLATNESS,
        CONTROLF=FINAL_LNF,
        q=Q,
        LB=LB,
        UB=UB,
    )

    #######################################################
    ############   Saving Data to a txt file   ############
    data_dict = {"E": REF, "lng(E)": LNGE_A, "H(E)": HIST_A}
    data = pd.DataFrame.from_dict(data_dict)
    data.to_csv(f"{DIRECTORY_NAME}/out_final.txt")
    print("Saved results.")
    #######################################################


if __name__ == "__main__":
    main()
