"""Simulated annealing v4: directed proposals, cheaper bookkeeping, smaller rematching.

Same idea and output files as annealing_v3.py, with these changes:
  * --p-dir: for a misplaced cube, the command is often chosen among those that reduce the
    distance to its target. The effect of each command is learned automatically at start-up;
    if the commands do not behave like fixed steps, this feature switches itself off.
  * Cheap cube selection: no O(n) numpy work and no rng.choice on every trial.
  * Rematching only involves cubes that are not yet correct (correct cubes stay where they are,
    which is optimal by the triangle inequality), so it gets much faster as C grows.
  * Python's random module for scalar draws (much faster than numpy's Generator).
  * --reheat-after N (off by default): if the best objective has not improved for N trials,
    the temperature is raised to --reheat-t and then decays back to the normal schedule.
Results go to *_annealing_v4*.json and results/annealing_v4_runs.csv (v3 files are untouched).
Seeds do not reproduce v3 runs because the random stream is different.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes

try:
    import msvcrt  # Windows

    def acquire_lock(file_handle):
        file_handle.seek(0)
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)

    def release_lock(file_handle):
        file_handle.seek(0)
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
except ImportError:
    import fcntl  # Linux (for example, Kaggle)

    def acquire_lock(file_handle):
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)

    def release_lock(file_handle):
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem", nargs="?", default="ISS")
    parser.add_argument("--trials", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t0", type=float, default=0.5)
    parser.add_argument("--t1", type=float, default=0.25)
    parser.add_argument("--start", default=None, help="JSON solution chromosome to resume from")
    parser.add_argument("--p-mis", type=float, default=0.7,
                        help="probability of selecting a misplaced cube")
    parser.add_argument("--p-near", type=float, default=0.0,
                        help="probability of selecting a cube near a misplaced cube/target")
    parser.add_argument("--p-dir", type=float, default=0.6,
                        help="probability of choosing a distance-reducing command for a misplaced cube")
    parser.add_argument("--p-zero", type=float, default=0.3,
                        help="probability of accepting a neutral move (lower it to cut wasted commands)")
    parser.add_argument("--radius", type=int, default=2,
                        help="Chebyshev radius for selecting nearby cubes")
    parser.add_argument("--rematch-every", type=int, default=200)
    parser.add_argument("--reheat-after", type=int, default=0,
                        help="reheat after this many trials without a new best (0 = off)")
    parser.add_argument("--reheat-t", type=float, default=None,
                        help="peak temperature of a reheat (default: sqrt(t0 * t1), the middle of the schedule)")
    parser.add_argument("--tag", default="v4", help="solution filename label")
    args = parser.parse_args()

    if args.trials <= 0 or args.seed < 0:
        parser.error("--trials must be positive and --seed nonnegative")
    if args.t0 <= 0 or args.t1 <= 0:
        parser.error("--t0 and --t1 must be positive")
    if args.p_mis < 0 or args.p_near < 0 or args.p_mis + args.p_near > 1:
        parser.error("--p-mis and --p-near must be nonnegative and sum to at most 1")
    if not 0 <= args.p_zero <= 1 or not 0 <= args.p_dir <= 1:
        parser.error("--p-zero and --p-dir must be between 0 and 1")
    if args.radius < 0 or args.rematch_every <= 0:
        parser.error("--radius must be nonnegative and --rematch-every positive")
    if args.reheat_after < 0 or (args.reheat_t is not None and args.reheat_t <= 0):
        parser.error("--reheat-after must be nonnegative and --reheat-t positive")
    return args


def main():
    args = parse_args()
    rnd = random.Random(args.seed)
    udp = programmable_cubes_UDP(args.problem)
    setup = udp.setup
    max_commands = setup["max_cmds"]
    initial_positions = np.load(f"{setup['path']}/Initial_Config.npy")
    types = udp.initial_cube_types
    target_positions = udp.target_cube_positions
    target_types = udp.target_cube_types
    n = len(initial_positions)
    num_types = setup["num_cube_types"]
    alpha = 0.1
    target_type_at = {
        tuple(position): int(cube_type)
        for position, cube_type in zip(target_positions.tolist(), target_types.tolist())
    }
    cube_worth = (1 - alpha) / n / (alpha / max_commands)
    print(f"max_cmds={max_commands} | one correct cube is worth about {cube_worth:.0f} commands")

    # Per-type data for the rematching.
    type_cubes = [np.where(types == t)[0] for t in range(num_types)]
    type_target_pos = [target_positions[target_types == t] for t in range(num_types)]
    type_target_tuples = [[tuple(p) for p in pos.tolist()] for pos in type_target_pos]

    def objective(correct, command_count):
        return (1 - alpha) * correct / n + alpha * (1 - command_count / max_commands)

    def learn_effects(probe_trials=3000):
        """Displacement of each command, if it is the same fixed step every time; else None."""
        probe = ProgrammableCubes(initial_positions)
        probe_rnd = random.Random(12345)
        seen = [dict() for _ in range(6)]
        for _ in range(probe_trials):
            cube_id = probe_rnd.randrange(n)
            command = probe_rnd.randrange(6)
            before = probe.cube_position[cube_id].copy()
            if probe.apply_single_update_step(cube_id, command) == 0:
                continue
            step = tuple((probe.cube_position[cube_id] - before).tolist())
            seen[command][step] = seen[command].get(step, 0) + 1
        effects = []
        for command in range(6):
            total = sum(seen[command].values())
            if total == 0:
                return None
            step, count = max(seen[command].items(), key=lambda item: item[1])
            if count < 0.98 * total or not any(step):
                return None
            effects.append(step)
        return effects

    effects = learn_effects() if args.p_dir > 0 else None
    print("directed proposals:", f"on, command steps {effects}" if effects else "off")

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

    def rematch(positions, flags):
        """Optimal L1 matching of the not-yet-correct cubes to the free targets of their type.

        Correct cubes keep their own position as target (distance 0), which is optimal."""
        assignment = positions.copy()
        for cube_type in range(num_types):
            all_idx = type_cubes[cube_type]
            wrong = ~flags[all_idx]
            idx = all_idx[wrong]
            if len(idx) == 0:
                continue
            occupied = {tuple(p) for p in positions[all_idx[~wrong]].tolist()}
            free = [k for k, tp in enumerate(type_target_tuples[cube_type]) if tp not in occupied]
            if len(free) == len(idx):
                targets = type_target_pos[cube_type][free]
            else:  # type counts do not match: fall back to the full matching
                idx = all_idx
                targets = type_target_pos[cube_type]
            costs = np.abs(positions[idx][:, None, :] - targets[None, :, :]).sum(axis=2)
            rows, cols = linear_sum_assignment(costs)
            assignment[idx[rows]] = targets[cols]
        return assignment

    correct_flags = np.array([is_correct(i) for i in range(n)])
    correct_count = int(correct_flags.sum())
    target_of = rematch(cubes.cube_position, correct_flags)
    distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
    total_distance = int(distances.sum())
    misplaced_pool = np.flatnonzero(distances > 0).tolist()
    best_objective = objective(correct_count, len(moves))
    best_length = len(moves)
    best_correct = correct_count
    accepted = 0
    print(
        f"start: E={total_distance} | C={correct_count} | L={len(moves)} "
        f"| f={-((best_objective - setup['fitness_offset']) / (1 - setup['fitness_offset'])):.5f}"
    )

    def pick_cube():
        roll = rnd.random()
        if not misplaced_pool or roll >= args.p_mis + args.p_near:
            return rnd.randrange(n)
        reference_cube = misplaced_pool[rnd.randrange(len(misplaced_pool))]
        for _ in range(5):  # the pool can be slightly stale; skip cubes that are now fine
            if distances[reference_cube] > 0:
                break
            reference_cube = misplaced_pool[rnd.randrange(len(misplaced_pool))]
        if roll < args.p_mis:
            return reference_cube
        reference = (cubes.cube_position[reference_cube] if rnd.random() < 0.5
                     else target_of[reference_cube])
        nearby = np.flatnonzero(
            np.abs(cubes.cube_position - reference).max(axis=1) <= args.radius
        )
        return int(nearby[rnd.randrange(len(nearby))]) if len(nearby) else reference_cube

    def pick_command(cube_id):
        if effects is not None and distances[cube_id] > 0 and rnd.random() < args.p_dir:
            dx, dy, dz = (target_of[cube_id] - cubes.cube_position[cube_id]).tolist()
            base = abs(dx) + abs(dy) + abs(dz)
            good = [c for c, (ex, ey, ez) in enumerate(effects)
                    if abs(dx - ex) + abs(dy - ey) + abs(dz - ez) < base]
            if good:
                return good[rnd.randrange(len(good))]
        return rnd.randrange(6)

    start_time = time.time()
    report_every = max(1, args.trials // 20)
    log_ratio = math.log(args.t1 / args.t0)
    reheat_peak = args.reheat_t if args.reheat_t is not None else math.sqrt(args.t0 * args.t1)
    reheat_half_life = max(1, args.reheat_after // 4)
    last_gain_trial = 0
    reheat_start = -1
    reheat_count = 0
    for trial in range(args.trials):
        temperature = args.t0 * math.exp(log_ratio * trial / args.trials)
        if args.reheat_after > 0:
            if (trial - max(last_gain_trial, reheat_start) >= args.reheat_after
                    and reheat_peak > temperature):
                reheat_start = trial
                reheat_count += 1
            if reheat_start >= 0:
                temperature = max(
                    temperature,
                    reheat_peak * 0.5 ** ((trial - reheat_start) / reheat_half_life),
                )
        if trial % report_every == 0:
            print(
                f"k={trial:7d} T={temperature:.3f} E={total_distance:5d} "
                f"C={correct_count:4d} L={len(moves):6d} "
                f"| best: C={best_correct} L={best_length}"
                + (f" | reheats={reheat_count}" if args.reheat_after > 0 else "")
            )
        if len(moves) >= max_commands - 2:
            print("command budget reached at trial", trial)
            break
        cube_id = pick_cube()
        command = pick_command(cube_id)
        old_position = cubes.cube_position[cube_id].copy()
        old_distance = int(distances[cube_id])
        if cubes.apply_single_update_step(cube_id, command) == 0:
            continue

        new_distance = int(np.abs(cubes.cube_position[cube_id] - target_of[cube_id]).sum())
        delta = new_distance - old_distance
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rnd.random() < args.p_zero
        else:
            accept = rnd.random() < math.exp(-delta / temperature)

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
            last_gain_trial = trial

        accepted += 1
        if accepted % args.rematch_every == 0:
            target_of = rematch(cubes.cube_position, correct_flags)
            distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
            total_distance = int(distances.sum())
            misplaced_pool = np.flatnonzero(distances > 0).tolist()

    elapsed = time.time() - start_time
    moves = moves[:best_length]
    chromosome = np.array([value for move in moves for value in move] + [-1], dtype=int)
    fitness = udp.fitness(chromosome)[0]
    final_positions = udp.final_cube_positions
    correct_final = 0
    for cube_type in range(num_types):
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
    best_path = f"solutions/{args.problem.lower()}_annealing_{args.tag}.json"
    best_lock_path = best_path + ".lock"
    with open(best_lock_path, "a+b") as lock_file:
        if os.path.getsize(best_lock_path) == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        acquire_lock(lock_file)
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
            release_lock(lock_file)

    csv_path = "results/annealing_v4_runs.csv"
    headers = ["problem", "tag", "seed", "t0", "t1", "trials", "p_mis", "p_near", "p_dir",
               "p_zero", "radius", "rematch_every", "start", "fitness", "C", "L", "seconds"]
    # Use a file lock so simultaneous runs append a single header and intact rows.
    with open(csv_path + ".lock", "a+b") as lock_file:
        if os.path.getsize(csv_path + ".lock") == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        acquire_lock(lock_file)
        try:
            with open(csv_path, "a+", newline="", encoding="utf-8") as csv_file:
                csv_file.seek(0, os.SEEK_END)
                is_empty = csv_file.tell() == 0
                writer = csv.writer(csv_file)
                if is_empty:
                    writer.writerow(headers)
                writer.writerow([
                    args.problem, args.tag, args.seed, args.t0, args.t1, args.trials,
                    args.p_mis, args.p_near, args.p_dir, args.p_zero, args.radius,
                    args.rematch_every, args.start or "", round(fitness, 5),
                    int(correct_final), len(moves), round(elapsed, 1),
                ])
                csv_file.flush()
        finally:
            lock_file.seek(0)
            release_lock(lock_file)
    print("logged to", csv_path)


if __name__ == "__main__":
    main()
