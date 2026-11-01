"""Simulated annealing with periodic rematching for Programmable Cubes.

Run from the project directory: python annealing_v2.py ISS [trials] [seed]
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ISS"
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # Annealing parameters. Temperature is measured in L1 distance units.
    temperature_start, temperature_end = 0.5, 0.25
    probability_zero = 0.3
    probability_misplaced = 0.7
    rematch_every = 200
    alpha = 0.1

    rng = np.random.default_rng(seed)
    udp = programmable_cubes_UDP(name)
    setup = udp.setup
    max_commands = setup["max_cmds"]
    initial_positions = np.load(f"{setup['path']}/Initial_Config.npy")
    types = udp.initial_cube_types
    target_positions = udp.target_cube_positions
    target_types = udp.target_cube_types
    n = len(initial_positions)

    target_type_at = {
        tuple(position): int(cube_type)
        for position, cube_type in zip(target_positions.tolist(), target_types.tolist())
    }

    def objective(correct, command_count):
        return (1 - alpha) * correct / n + alpha * (1 - command_count / max_commands)

    def rematch(positions):
        """Assign each cube a target of its type using minimum total L1 distance."""
        target_of = np.zeros_like(positions)
        for cube_type in range(setup["num_cube_types"]):
            indices = np.where(types == cube_type)[0]
            targets = target_positions[target_types == cube_type]
            costs = np.abs(positions[indices][:, None, :] - targets[None, :, :]).sum(axis=2)
            rows, cols = linear_sum_assignment(costs)
            target_of[indices[rows]] = targets[cols]
        return target_of

    cubes = ProgrammableCubes(initial_positions)

    def is_correct(cube_id):
        position = tuple(cubes.cube_position[cube_id].tolist())
        return target_type_at.get(position) == int(types[cube_id])

    target_of = rematch(cubes.cube_position)
    distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
    total_distance = int(distances.sum())
    correct_flags = np.array([is_correct(i) for i in range(n)])
    correct_count = int(correct_flags.sum())

    moves = []
    best_objective = objective(correct_count, 0)
    best_length = 0
    best_correct = correct_count
    accepted = 0
    print(
        "start: E =", total_distance,
        "| C =", correct_count,
        "| F0 =", round(best_objective, 5),
        "| F0 zyrtar =", round(setup["fitness_offset"], 5),
    )

    start_time = time.time()
    report_every = max(1, trials // 20)
    for trial in range(trials):
        if trial % report_every == 0:
            temperature = temperature_start * (temperature_end / temperature_start) ** (trial / trials)
            print(
                f"k={trial:6d} T={temperature:.3f} E={total_distance:5d} "
                f"C={correct_count:3d} L={len(moves):5d} | "
                f"më e mira: C={best_correct} L={best_length}"
            )
        if len(moves) >= max_commands - 2:
            break
        temperature = temperature_start * (temperature_end / temperature_start) ** (trial / trials)

        if rng.random() < probability_misplaced:
            misplaced = np.flatnonzero(distances > 0)
            cube_id = int(rng.choice(misplaced)) if len(misplaced) else int(rng.integers(n))
        else:
            cube_id = int(rng.integers(n))
        command = int(rng.integers(6))

        old_position = cubes.cube_position[cube_id].copy()
        old_distance = int(distances[cube_id])
        if cubes.apply_single_update_step(cube_id, command) == 0:
            continue

        new_distance = int(np.abs(cubes.cube_position[cube_id] - target_of[cube_id]).sum())
        delta = new_distance - old_distance
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < probability_zero
        else:
            accept = rng.random() < math.exp(-delta / temperature)

        if accept:
            moves.append((cube_id, command))
        else:
            undo_ok = cubes.apply_single_update_step(cube_id, command ^ 1)
            if undo_ok == 1 and np.array_equal(cubes.cube_position[cube_id], old_position):
                continue
            # Keep the chromosome consistent if the trial could not be undone.
            moves.append((cube_id, command))
            if undo_ok == 1:
                moves.append((cube_id, command ^ 1))

        current_distance = int(np.abs(cubes.cube_position[cube_id] - target_of[cube_id]).sum())
        distances[cube_id] = current_distance
        total_distance += current_distance - old_distance
        current_correct = is_correct(cube_id)
        correct_count += int(current_correct) - int(correct_flags[cube_id])
        correct_flags[cube_id] = current_correct

        current_objective = objective(correct_count, len(moves))
        if current_objective > best_objective:
            best_objective = current_objective
            best_length = len(moves)
            best_correct = correct_count

        accepted += 1
        if accepted % rematch_every == 0:
            target_of = rematch(cubes.cube_position)
            distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
            total_distance = int(distances.sum())

    elapsed = time.time() - start_time

    # Evaluate the best prefix from the initial configuration with the official fitness.
    moves = moves[:best_length]
    chromosome = np.array([value for move in moves for value in move] + [-1], dtype=int)
    fitness = udp.fitness(chromosome)[0]
    final_positions = udp.final_cube_positions
    correct_final = 0
    for cube_type in range(setup["num_cube_types"]):
        final_set = {tuple(position) for position in final_positions[types == cube_type].tolist()}
        correct_final += sum(
            tuple(position) in final_set
            for position in target_positions[target_types == cube_type].tolist()
        )

    print(f"\n=== REZULTATI v2 ({name}) ===")
    print(f"fitness f: {fitness:.5f} (sa më negativ aq më mirë, min teorik -1)")
    print(
        f"C = {correct_final} nga {n} | L = {len(moves)} komanda | "
        f"koha: {elapsed:.1f} s | seed: {seed}"
    )
    if correct_final != best_correct:
        print(
            f"KUJDES: C i kërkimit ({best_correct}) ndryshon "
            f"nga C zyrtar ({correct_final})"
        )

    v1_path = f"solutions/{name.lower()}_greedy_v1.json"
    if os.path.exists(v1_path):
        with open(v1_path, encoding="utf-8") as input_file:
            v1 = json.load(input_file)
        print(
            f"v1 (greedy): f = {v1['fitness']:.5f} | "
            f"C = {v1['C']} | L = {v1['L']}"
        )

    os.makedirs("solutions", exist_ok=True)
    output_path = f"solutions/{name.lower()}_annealing_v2_seed_{seed}.json"
    better = True
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as input_file:
            previous = json.load(input_file)
        better = fitness < previous["fitness"]

    if better:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "problem": name,
                    "fitness": fitness,
                    "C": int(correct_final),
                    "L": len(moves),
                    "trials": trials,
                    "seed": seed,
                    "T0": temperature_start,
                    "T1": temperature_end,
                    "chromosome": [int(value) for move in moves for value in move] + [-1],
                },
                output_file,
                indent=2,
            )
        print("ruajtur si rezultati më i mirë te", output_path)
    else:
        print("nuk është më i mirë se rezultati i ruajtur; skedari nuk u mbishkrua")


if __name__ == "__main__":
    main()
