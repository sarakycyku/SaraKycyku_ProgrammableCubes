"""Simulated annealing with CLI parameters, restarts and blocker-aware sampling.

Run from the project directory; see --help for options.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", nargs="?", default="ISS")
    parser.add_argument("--trials", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t0", type=float, default=0.5)
    parser.add_argument("--t1", type=float, default=0.25)
    parser.add_argument("--start", default=None, help="JSON solution chromosome to resume from")
    parser.add_argument("--p-mis", type=float, default=0.5,
                        help="probability of selecting a misplaced cube")
    parser.add_argument("--p-near", type=float, default=0.3,
                        help="probability of selecting a cube near a misplaced cube/target")
    parser.add_argument("--p-zero", type=float, default=0.3,
                        help="probability of accepting a neutral move")
    parser.add_argument("--radius", type=int, default=2,
                        help="Chebyshev radius for selecting nearby cubes")
    parser.add_argument("--rematch-every", type=int, default=200)
    parser.add_argument("--tag", default="v3", help="solution filename label")
    args = parser.parse_args()

    if args.trials <= 0 or args.seed < 0:
        parser.error("--trials must be positive and --seed nonnegative")
    if args.t0 <= 0 or args.t1 <= 0:
        parser.error("--t0 and --t1 must be positive")
    if args.p_mis < 0 or args.p_near < 0 or args.p_mis + args.p_near > 1:
        parser.error("--p-mis and --p-near must be nonnegative and sum to at most 1")
    if not 0 <= args.p_zero <= 1:
        parser.error("--p-zero must be between 0 and 1")
    if args.radius < 0 or args.rematch_every <= 0:
        parser.error("--radius must be nonnegative and --rematch-every positive")
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    udp = programmable_cubes_UDP(args.problem)
    setup = udp.setup
    max_commands = setup["max_cmds"]
    initial_positions = np.load(f"{setup['path']}/Initial_Config.npy")
    types = udp.initial_cube_types
    target_positions = udp.target_cube_positions
    target_types = udp.target_cube_types
    n = len(initial_positions)
    alpha = 0.1
    target_type_at = {
        tuple(position): int(cube_type)
        for position, cube_type in zip(target_positions.tolist(), target_types.tolist())
    }

    def objective(correct, command_count):
        return (1 - alpha) * correct / n + alpha * (1 - command_count / max_commands)

    def rematch(positions):
        assignment = np.zeros_like(positions)
        for cube_type in range(setup["num_cube_types"]):
            indices = np.where(types == cube_type)[0]
            targets = target_positions[target_types == cube_type]
            costs = np.abs(positions[indices][:, None, :] - targets[None, :, :]).sum(axis=2)
            rows, cols = linear_sum_assignment(costs)
            assignment[indices[rows]] = targets[cols]
        return assignment

    cubes = ProgrammableCubes(initial_positions)
    moves = []
    start_fitness = None
    if args.start:
        with open(args.start, encoding="utf-8") as source:
            start_data = json.load(source)
        chromosome = start_data["chromosome"]
        start_fitness = start_data.get("fitness")
        illegal_commands = 0
        for index in range(0, len(chromosome) - 1, 2):
            if chromosome[index] == -1:
                break
            cube_id, command = int(chromosome[index]), int(chromosome[index + 1])
            moves.append((cube_id, command))
            if cubes.apply_single_update_step(cube_id, command) == 0:
                illegal_commands += 1
        print(f"start: loaded {len(moves)} commands from {args.start} | illegal: {illegal_commands}")

    def is_correct(cube_id):
        position = tuple(cubes.cube_position[cube_id].tolist())
        return target_type_at.get(position) == int(types[cube_id])

    target_of = rematch(cubes.cube_position)
    distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
    total_distance = int(distances.sum())
    correct_flags = np.array([is_correct(i) for i in range(n)])
    correct_count = int(correct_flags.sum())
    best_objective = objective(correct_count, len(moves))
    best_length = len(moves)
    best_correct = correct_count
    accepted = 0
    print(
        f"start: E={total_distance} | C={correct_count} | L={len(moves)} "
        f"| f={-((best_objective - setup['fitness_offset']) / (1 - setup['fitness_offset'])):.5f}"
    )

    def pick_cube():
        roll = rng.random()
        misplaced = np.flatnonzero(distances > 0)
        if len(misplaced) == 0 or roll >= args.p_mis + args.p_near:
            return int(rng.integers(n))
        reference_cube = int(rng.choice(misplaced))
        if roll < args.p_mis:
            return reference_cube
        reference = (cubes.cube_position[reference_cube] if rng.random() < 0.5
                     else target_of[reference_cube])
        nearby = np.flatnonzero(
            np.abs(cubes.cube_position - reference).max(axis=1) <= args.radius
        )
        return int(rng.choice(nearby)) if len(nearby) else reference_cube

    start_time = time.time()
    report_every = max(1, args.trials // 20)
    for trial in range(args.trials):
        if trial % report_every == 0:
            temperature = args.t0 * (args.t1 / args.t0) ** (trial / args.trials)
            print(
                f"k={trial:7d} T={temperature:.3f} E={total_distance:5d} "
                f"C={correct_count:4d} L={len(moves):6d} "
                f"| best: C={best_correct} L={best_length}"
            )
        if len(moves) >= max_commands - 2:
            print("command budget reached at trial", trial)
            break
        temperature = args.t0 * (args.t1 / args.t0) ** (trial / args.trials)
        cube_id = pick_cube()
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
            accept = rng.random() < args.p_zero
        else:
            accept = rng.random() < math.exp(-delta / temperature)

        if accept:
            moves.append((cube_id, command))
        else:
            undo_ok = cubes.apply_single_update_step(cube_id, command ^ 1)
            if undo_ok == 1 and np.array_equal(cubes.cube_position[cube_id], old_position):
                continue
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
            best_objective, best_length, best_correct = current_objective, len(moves), correct_count

        accepted += 1
        if accepted % args.rematch_every == 0:
            target_of = rematch(cubes.cube_position)
            distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
            total_distance = int(distances.sum())

    elapsed = time.time() - start_time
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

    print(f"\n=== RESULT {args.tag} ({args.problem}) ===")
    print(
        f"f={fitness:.5f} | C={correct_final}/{n} | L={len(moves)} "
        f"| time={elapsed:.1f}s | seed={args.seed}"
    )
    if correct_final != best_correct:
        print(f"WARNING: tracked C={best_correct} differs from official C={correct_final}")
    if start_fitness is not None:
        print(f"start solution f={start_fitness:.5f}")

    os.makedirs("solutions", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    # Each run gets its own file, allowing concurrent experiments without write races.
    output_path = f"solutions/{args.problem.lower()}_annealing_{args.tag}_seed_{args.seed}.json"
    run_data = {
        "problem": args.problem,
        "fitness": fitness,
        "C": int(correct_final),
        "L": len(moves),
        "args": vars(args),
        "chromosome": [int(value) for move in moves for value in move] + [-1],
    }
    better = True
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as source:
            better = fitness < json.load(source)["fitness"]
    if better:
        with open(output_path, "w", encoding="utf-8") as destination:
            json.dump(run_data, destination, indent=2)
        print("saved", output_path)
    else:
        print("not better than the stored run with this seed; not overwritten")

    # Also maintain one best-so-far solution across seeds and experiment settings.
    import msvcrt
    best_path = f"solutions/{args.problem.lower()}_annealing_v3.json"
    best_lock_path = best_path + ".lock"
    with open(best_lock_path, "a+b") as lock_file:
        if os.path.getsize(best_lock_path) == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            existing_fitness = None
            if os.path.exists(best_path):
                with open(best_path, encoding="utf-8") as source:
                    existing_fitness = json.load(source)["fitness"]
            if existing_fitness is None or fitness < existing_fitness:
                with open(best_path, "w", encoding="utf-8") as destination:
                    json.dump(run_data, destination, indent=2)
                print("updated best", best_path)
            else:
                print("best solution unchanged", best_path)
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    csv_path = "results/annealing_v3_runs.csv"
    headers = ["problem", "tag", "seed", "t0", "t1", "trials", "p_mis", "p_near",
               "p_zero", "radius", "rematch_every", "start", "fitness", "C", "L", "seconds"]
    # Use a file lock so simultaneous runs append a single header and intact rows.
    with open(csv_path + ".lock", "a+b") as lock_file:
        if os.path.getsize(csv_path + ".lock") == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            with open(csv_path, "a+", newline="", encoding="utf-8") as csv_file:
                csv_file.seek(0, os.SEEK_END)
                is_empty = csv_file.tell() == 0
                writer = csv.writer(csv_file)
                if is_empty:
                    writer.writerow(headers)
                writer.writerow([
                    args.problem, args.tag, args.seed, args.t0, args.t1, args.trials,
                    args.p_mis, args.p_near, args.p_zero, args.radius, args.rematch_every,
                    args.start or "", round(fitness, 5), int(correct_final), len(moves), round(elapsed, 1),
                ])
                csv_file.flush()
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    print("logged to", csv_path)


if __name__ == "__main__":
    main()
