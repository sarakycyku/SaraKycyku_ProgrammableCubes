"""Greedy baseline using the official Programmable Cubes simulator.

Run from the project directory: python baseline.py ISS
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ISS"
    udp = programmable_cubes_UDP(name)
    setup = udp.setup
    max_cmds = setup["max_cmds"]

    initial_positions = np.load(f"{setup['path']}/Initial_Config.npy")
    types = udp.initial_cube_types
    target_positions = udp.target_cube_positions
    target_types = udp.target_cube_types

    def count_correct(positions):
        """Count cubes at target positions with matching types."""
        total = 0
        for cube_type in range(setup["num_cube_types"]):
            final = {tuple(p) for p in positions[types == cube_type].tolist()}
            total += sum(
                tuple(p) in final
                for p in target_positions[target_types == cube_type].tolist()
            )
        return total

    def evaluate(moves):
        chromosome = [value for move in moves for value in move] + [-1]
        return udp.fitness(np.asarray(chromosome, dtype=int))[0]

    # Assign each cube a target of the same type, minimizing total L1 distance.
    target_of = np.zeros_like(initial_positions)
    for cube_type in range(setup["num_cube_types"]):
        indices = np.where(types == cube_type)[0]
        targets = target_positions[target_types == cube_type]
        costs = np.abs(initial_positions[indices][:, None, :] - targets[None, :, :]).sum(axis=2)
        rows, cols = linear_sum_assignment(costs)
        target_of[indices[rows]] = targets[cols]

    start_time = time.time()
    cubes = ProgrammableCubes(initial_positions)
    moves = []

    def distance(cube_id):
        return int(np.abs(cubes.cube_position[cube_id] - target_of[cube_id]).sum())

    improved = True
    passes = 0
    while improved and len(moves) < max_cmds:
        improved = False
        passes += 1
        order = sorted(
            (i for i in range(len(initial_positions)) if distance(i) > 0),
            key=distance,
            reverse=True,
        )
        for cube_id in order:
            if len(moves) >= max_cmds - 2:
                break
            for command in range(6):
                old_distance = distance(cube_id)
                if old_distance == 0:
                    break
                old_position = cubes.cube_position[cube_id].copy()
                if cubes.apply_single_update_step(cube_id, command) == 0:
                    continue
                if distance(cube_id) < old_distance:
                    moves.append((cube_id, command))
                    improved = True
                    break

                # Discard non-improving trials by applying the inverse command.
                undo_ok = cubes.apply_single_update_step(cube_id, command ^ 1)
                if undo_ok == 1 and np.array_equal(cubes.cube_position[cube_id], old_position):
                    continue

                # Record any commands that actually changed the state.
                moves.append((cube_id, command))
                if undo_ok == 1:
                    moves.append((cube_id, command ^ 1))
        print(
            f"kalim {passes} | lëvizje: {len(moves)} | "
            f"kube në vend (L1=0): {sum(distance(i) == 0 for i in range(len(initial_positions)))}"
        )

    moves = moves[:max_cmds]
    elapsed = time.time() - start_time

    fitness = evaluate(moves)
    correct = count_correct(udp.final_cube_positions)
    print(f"\n=== REZULTATI ({name}) ===")
    print(f"fitness f: {fitness:.5f} (sa më negativ aq më mirë, min teorik -1)")
    print(
        f"C = {correct} nga {len(initial_positions)} | L = {len(moves)} komanda | "
        f"koha: {elapsed:.1f} s"
    )

    os.makedirs("solutions", exist_ok=True)
    output_path = f"solutions/{name.lower()}_greedy_v1.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "problem": name,
                "fitness": fitness,
                "C": int(correct),
                "L": len(moves),
                "chromosome": [int(value) for move in moves for value in move] + [-1],
            },
            output_file,
            indent=2,
        )
    print("ruajtur te", output_path)


if __name__ == "__main__":
    main()
