"""Tabu search for the Programmable Cubes problems (ISS, JWST, Enterprise).

Compared with annealing_v4.py there is no temperature. In every iteration a small sample of
candidate moves (cube, command) is evaluated and the BEST non-tabu candidate is applied, even
if it makes things worse. To avoid cycling, the cube that just left a position may not return
to that position for a random number of iterations (the tabu list, keyed by cube and position).
Aspiration: a tabu move is still allowed if it puts a misplaced cube exactly on its target.

Same output conventions as annealing_v4.py:
  solutions/<problem>_tabu_<tag>_seed_<seed>.json, solutions/<problem>_tabu_<tag>.json (best),
  results/tabu_runs.csv. Use --start to continue from a saved solution.

Example:  python tabu_search.py ISS --iters 200000 --seed 0
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes
from annealing_v4 import acquire_lock, release_lock


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem", nargs="?", default="ISS")
    parser.add_argument("--iters", type=int, default=200000,
                        help="tabu iterations (each applies one move)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8,
                        help="candidate moves evaluated per iteration")
    parser.add_argument("--tenure", type=int, default=30,
                        help="base tabu tenure; the real one is random in [tenure, 2*tenure]")
    parser.add_argument("--p-mis", type=float, default=0.7,
                        help="probability that a candidate cube is a misplaced one")
    parser.add_argument("--p-dir", type=float, default=0.5,
                        help="probability that a candidate command reduces the distance")
    parser.add_argument("--rematch-every", type=int, default=200)
    parser.add_argument("--start", default=None, help="JSON solution chromosome to resume from")
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    if args.iters <= 0 or args.seed < 0 or args.candidates <= 0 or args.tenure < 0:
        parser.error("--iters/--candidates must be positive; --seed/--tenure nonnegative")
    if not 0 <= args.p_mis <= 1 or not 0 <= args.p_dir <= 1:
        parser.error("--p-mis and --p-dir must be between 0 and 1")
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
    target_type_at = {tuple(p): int(t) for p, t in zip(target_positions.tolist(),
                                                       target_types.tolist())}
    type_cubes = [np.where(types == t)[0] for t in range(num_types)]
    type_target_pos = [target_positions[target_types == t] for t in range(num_types)]
    type_target_tuples = [[tuple(p) for p in pos.tolist()] for pos in type_target_pos]

    def objective(correct, command_count):
        return (1 - alpha) * correct / n + alpha * (1 - command_count / max_commands)

    def learn_effects(probe_trials=3000):
        probe = ProgrammableCubes(initial_positions)
        probe_rnd = random.Random(12345)
        seen = [dict() for _ in range(6)]
        for _ in range(probe_trials):
            cube_id, command = probe_rnd.randrange(n), probe_rnd.randrange(6)
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
    print("directed candidates:", f"on, steps {effects}" if effects else "off")

    cubes = ProgrammableCubes(initial_positions)
    moves = []
    if args.start:
        with open(args.start, encoding="utf-8") as source:
            chromosome = json.load(source)["chromosome"]
        illegal = 0
        for index in range(0, len(chromosome) - 1, 2):
            if chromosome[index] == -1:
                break
            cube_id, command = int(chromosome[index]), int(chromosome[index + 1])
            moves.append((cube_id, command))
            if cubes.apply_single_update_step(cube_id, command) == 0:
                illegal += 1
        print(f"start: loaded {len(moves)} commands | illegal: {illegal}")

    def is_correct(cube_id):
        return target_type_at.get(tuple(cubes.cube_position[cube_id].tolist())) == int(types[cube_id])

    def rematch(positions, flags):
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
            else:
                idx, targets = all_idx, type_target_pos[cube_type]
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
    best_length, best_correct = len(moves), correct_count
    print(f"start: E={total_distance} | C={correct_count} | L={len(moves)}")

    tabu = {}  # (cube, position tuple) -> iteration until which it is forbidden

    def pick_candidate():
        if misplaced_pool and rnd.random() < args.p_mis:
            cube_id = misplaced_pool[rnd.randrange(len(misplaced_pool))]
        else:
            cube_id = rnd.randrange(n)
        if effects is not None and distances[cube_id] > 0 and rnd.random() < args.p_dir:
            dx, dy, dz = (target_of[cube_id] - cubes.cube_position[cube_id]).tolist()
            base = abs(dx) + abs(dy) + abs(dz)
            good = [c for c, (ex, ey, ez) in enumerate(effects)
                    if abs(dx - ex) + abs(dy - ey) + abs(dz - ez) < base]
            if good:
                return cube_id, good[rnd.randrange(len(good))]
        return cube_id, rnd.randrange(6)

    def update_bookkeeping(cube_id, old_distance):
        nonlocal total_distance, correct_count, best_objective, best_length, best_correct
        current = int(np.abs(cubes.cube_position[cube_id] - target_of[cube_id]).sum())
        distances[cube_id] = current
        total_distance += current - old_distance
        flag = is_correct(cube_id)
        correct_count += int(flag) - int(correct_flags[cube_id])
        correct_flags[cube_id] = flag
        value = objective(correct_count, len(moves))
        if value > best_objective:
            best_objective, best_length, best_correct = value, len(moves), correct_count

    start_time = time.time()
    report_every = max(1, args.iters // 20)
    applied = 0
    for iteration in range(args.iters):
        if iteration % report_every == 0:
            print(f"it={iteration:7d} E={total_distance:5d} C={correct_count:4d} L={len(moves):6d} "
                  f"| best: C={best_correct} L={best_length} | tabu={len(tabu)}")
            tabu = {k: v for k, v in tabu.items() if v > iteration}
        if len(moves) >= max_commands - 2:
            print("command budget reached at iteration", iteration)
            break

        best = None  # (delta, tiebreak, cube, command, new_position, aspirated)
        forced = False
        for _ in range(args.candidates):
            cube_id, command = pick_candidate()
            old_position = cubes.cube_position[cube_id].copy()
            old_distance = int(distances[cube_id])
            if cubes.apply_single_update_step(cube_id, command) == 0:
                continue
            new_position = tuple(cubes.cube_position[cube_id].tolist())
            new_distance = int(np.abs(cubes.cube_position[cube_id] - target_of[cube_id]).sum())
            undo_ok = cubes.apply_single_update_step(cube_id, command ^ 1)
            if not (undo_ok == 1 and np.array_equal(cubes.cube_position[cube_id], old_position)):
                # the probe could not be undone cleanly: it really happened, so record it
                moves.append((cube_id, command))
                if undo_ok == 1:
                    moves.append((cube_id, command ^ 1))
                update_bookkeeping(cube_id, old_distance)
                forced = True
                break
            is_tabu = tabu.get((cube_id, new_position), -1) > iteration
            aspirated = is_tabu and new_distance == 0 and old_distance > 0
            if is_tabu and not aspirated:
                continue
            key = (new_distance - old_distance, rnd.random(), cube_id, command, aspirated)
            if best is None or key < best:
                best = key
        if forced or best is None:
            continue

        _, _, cube_id, command, _ = best
        old_position = tuple(cubes.cube_position[cube_id].tolist())
        old_distance = int(distances[cube_id])
        if cubes.apply_single_update_step(cube_id, command) == 0:
            continue
        moves.append((cube_id, command))
        tabu[(cube_id, old_position)] = iteration + rnd.randint(args.tenure, 2 * args.tenure)
        update_bookkeeping(cube_id, old_distance)
        applied += 1
        if applied % args.rematch_every == 0:
            target_of = rematch(cubes.cube_position, correct_flags)
            distances = np.abs(cubes.cube_position - target_of).sum(axis=1)
            total_distance = int(distances.sum())
            misplaced_pool = np.flatnonzero(distances > 0).tolist()

    elapsed = time.time() - start_time
    moves = moves[:best_length]
    chromosome = np.array([v for move in moves for v in move] + [-1], dtype=int)
    fitness = udp.fitness(chromosome)[0]
    final_positions = udp.final_cube_positions
    correct_final = 0
    for cube_type in range(num_types):
        final_set = {tuple(p) for p in final_positions[types == cube_type].tolist()}
        correct_final += sum(tuple(p) in final_set for p in
                             target_positions[target_types == cube_type].tolist())
    print(f"\n=== RESULT tabu {args.tag} ({args.problem}) ===")
    print(f"f={fitness:.5f} | C={correct_final}/{n} | L={len(moves)} | time={elapsed:.1f}s | seed={args.seed}")
    if correct_final != best_correct:
        print(f"WARNING: tracked C={best_correct} differs from official C={correct_final}")

    os.makedirs("solutions", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    run_data = {"problem": args.problem, "fitness": fitness, "C": int(correct_final),
                "L": len(moves), "args": vars(args),
                "chromosome": [int(v) for move in moves for v in move] + [-1]}
    seed_path = f"solutions/{args.problem.lower()}_tabu_{args.tag}_seed_{args.seed}.json"
    best_path = f"solutions/{args.problem.lower()}_tabu_{args.tag}.json"
    if not os.path.exists(seed_path) or fitness < json.load(open(seed_path, encoding="utf-8"))["fitness"]:
        with open(seed_path, "w", encoding="utf-8") as handle:
            json.dump(run_data, handle, indent=2)
        print("saved", seed_path)

    def locked(path, action):
        with open(path + ".lock", "a+b") as lock_file:
            if os.path.getsize(path + ".lock") == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            acquire_lock(lock_file)
            try:
                action()
            finally:
                lock_file.seek(0)
                release_lock(lock_file)

    def update_best():
        if os.path.exists(best_path):
            with open(best_path, encoding="utf-8") as source:
                if fitness >= json.load(source)["fitness"]:
                    print("best solution unchanged", best_path)
                    return
        with open(best_path, "w", encoding="utf-8") as handle:
            json.dump(run_data, handle, indent=2)
        print("updated best", best_path)

    def append_csv():
        csv_path = "results/tabu_runs.csv"
        is_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if is_empty:
                writer.writerow(["problem", "tag", "seed", "iters", "candidates", "tenure",
                                 "p_mis", "p_dir", "start", "fitness", "C", "L", "seconds"])
            writer.writerow([args.problem, args.tag, args.seed, args.iters, args.candidates,
                             args.tenure, args.p_mis, args.p_dir, args.start or "",
                             round(fitness, 5), int(correct_final), len(moves), round(elapsed, 1)])
        print("logged to", csv_path)

    locked(best_path, update_best)
    locked("results/tabu_runs.csv", append_csv)


if __name__ == "__main__":
    main()
