"""Explore a Programmable Cubes scenario before designing a solver.

Run from the project directory, for example: python explore.py ISS
"""
import sys

import numpy as np

from programmable_cubes_UDP import programmable_cubes_UDP


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ISS"
    udp = programmable_cubes_UDP(name)
    setup = udp.setup

    initial_positions = np.load(f"{setup['path']}/Initial_Config.npy")
    initial_types = udp.initial_cube_types
    target_positions = udp.target_cube_positions
    target_types = udp.target_cube_types

    print(f"=== Problemi: {name}")
    print(
        f"pozicione fillestare: {len(initial_positions)} | "
        f"pozicione synimi: {len(target_positions)} | "
        f"lloje: {setup['num_cube_types']}"
    )

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None
        print("(scipy mungon; shuma L1 nuk do të llogaritet)")

    total_to_move = 0
    total_l1 = 0
    print("\nlloji | kube | në vend | duhen lëvizur | shuma L1 minimale")
    for cube_type in range(setup["num_cube_types"]):
        initial = initial_positions[initial_types == cube_type]
        target = target_positions[target_types == cube_type]
        initial_set = {tuple(position) for position in initial.tolist()}
        target_set = {tuple(position) for position in target.tolist()}
        in_place = len(initial_set & target_set)
        to_move = len(initial) - in_place
        l1 = "—"
        if linear_sum_assignment is not None:
            costs = np.abs(initial[:, None, :] - target[None, :, :]).sum(axis=2)
            rows, cols = linear_sum_assignment(costs)
            l1_value = int(costs[rows, cols].sum())
            total_l1 += l1_value
            l1 = str(l1_value)
        total_to_move += to_move
        print(f"{cube_type:5d} | {len(initial):4d} | {in_place:7d} | {to_move:13d} | {l1}")

    print(f"\nGjithsej duhen lëvizur: {total_to_move} nga {len(initial_positions)} kube.")
    if linear_sum_assignment is not None:
        print(f"Kufi i poshtëm L1/2 për lëvizjet: {int(np.ceil(total_l1 / 2))}")

    # An empty chromosome leaves the ensemble at its initial configuration.
    udp.fitness(np.array([-1], dtype=int))
    print("\nPo vizatoj konfigurimin fillestar dhe synimin; mbylli figurat për të vazhduar.")
    all_types = list(range(setup["num_cube_types"]))
    udp.plot("ensemble", all_types)
    udp.plot("target", all_types)


if __name__ == "__main__":
    main()
