"""Simulated annealing v5: v3-like search, much cheaper per trial.

What changed compared with annealing_v4.py (see the run statistics: v4's defaults were worse
than v3 on JWST, f mean -0.642 vs -0.685 at the same trial budget):
  * Defaults are back to the v3 baseline: --p-mis 0.5, --p-dir 0 (off), --p-zero 0.3.
    The directed proposals of v4 stay available (--p-dir) but are opt-in; use tune.py to test them.
  * Fast rejection: the new distance of a proposal is computed arithmetically from the learned
    step of the command, the Metropolis decision is taken FIRST, and the simulator is called only
    for accepted proposals. Rejected proposals cost no simulator call at all (v3/v4 spent two
    calls on every legal rejected proposal). The accepted-move statistics are unchanged, because
    legality does not depend on the random draw. Switched off automatically if the simulator's
    commands are not fixed steps, or with --no-fast-reject.
  * The set of misplaced cubes is kept exactly up to date (v4's pool was stale between rematches).
  * Rematching only involves the cubes that are not yet correct (equivalent, much cheaper).
  * Optional reheating (--reheat-after), as in v4.
Outputs: solutions/<problem>_annealing_<tag>_seed_<seed>.json, solutions/<problem>_annealing_<tag>.json
(best over seeds), results/annealing_v5_runs.csv. Use --start to continue from a saved solution.
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


# ----------------------------------------------------------------------------------------
# Helpers shared with tabu_search_v2.py
# ----------------------------------------------------------------------------------------
def learn_effects(initial_positions, probe_trials=3000):
    """Displacement of each of the 6 commands if it is always the same fixed step, else None."""
    n = len(initial_positions)
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
    return tuple(effects)


def make_rematch(types, target_positions, target_types, num_types):
    """Optimal L1 matching of the not-yet-correct cubes to the free targets of their type.

    Correct cubes keep their own position as target (distance 0), which is optimal."""
    type_cubes = [np.where(types == t)[0] for t in range(num_types)]
    type_target_pos = [target_positions[target_types == t] for t in range(num_types)]
    type_target_tuples = [[tuple(p) for p in pos.tolist()] for pos in type_target_pos]

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
            else:  # type counts do not match: fall back to the full matching
                idx = all_idx
                targets = type_target_pos[cube_type]
            costs = np.abs(positions[idx][:, None, :] - targets[None, :, :]).sum(axis=2)
            rows, cols = linear_sum_assignment(costs)
            assignment[idx[rows]] = targets[cols]
        return assignment

    return rematch


def official_result(udp, moves, types, target_positions, target_types, num_types):
    """Replay with the official simulator. Returns (fitness, number of correct cubes)."""
    chromosome = np.array([value for move in moves for value in move] + [-1], dtype=int)
    fitness = udp.fitness(chromosome)[0]
    final_positions = udp.final_cube_positions
    correct = 0
    for cube_type in range(num_types):
        final_set = {tuple(p) for p in final_positions[types == cube_type].tolist()}
        correct += sum(tuple(p) in final_set for p in
                       target_positions[target_types == cube_type].tolist())
    return fitness, correct


def locked(path, action):
    """Run action() while holding an exclusive lock stored next to path."""
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


def append_csv(csv_path, headers, row):
    def write():
        is_empty = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_empty:
                writer.writerow(headers)
            writer.writerow(row)
    locked(csv_path, write)
    print("logged to", csv_path)


def save_solution(prefix, problem, tag, seed, run_data):
    """Per-seed file (kept if better) and one best-so-far file across seeds."""
    os.makedirs("solutions", exist_ok=True)
    fitness = run_data["fitness"]
    seed_path = f"solutions/{problem.lower()}_{prefix}_{tag}_seed_{seed}.json"
    best_path = f"solutions/{problem.lower()}_{prefix}_{tag}.json"
    better = True
    if os.path.exists(seed_path):
        with open(seed_path, encoding="utf-8") as source:
            better = fitness < json.load(source)["fitness"]
    if better:
        with open(seed_path, "w", encoding="utf-8") as handle:
            json.dump(run_data, handle, indent=2)
        print("saved", seed_path)
    else:
        print("not better than the stored run with this seed; not overwritten")

    def update_best():
        if os.path.exists(best_path):
            with open(best_path, encoding="utf-8") as source:
                if fitness >= json.load(source)["fitness"]:
                    print("best solution unchanged", best_path)
                    return
        with open(best_path, "w", encoding="utf-8") as handle:
            json.dump(run_data, handle, indent=2)
        print("updated best", best_path)

    locked(best_path, update_best)


# ----------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem", nargs="?", default="ISS")
    parser.add_argument("--trials", type=int, default=600000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t0", type=float, default=0.5)
    parser.add_argument("--t1", type=float, default=0.25)
    parser.add_argument("--start", default=None, help="JSON solution chromosome to resume from")
    parser.add_argument("--p-mis", type=float, default=0.5,
                        help="probability of selecting a misplaced cube")
    parser.add_argument("--p-near", type=float, default=0.0,
                        help="probability of selecting a cube near a misplaced cube/target")
    parser.add_argument("--p-dir", type=float, default=0.0,
                        help="probability of choosing a distance-reducing command (0 = off)")
    parser.add_argument("--p-zero", type=float, default=0.3,
                        help="probability of accepting a neutral move")
    parser.add_argument("--radius", type=int, default=2,
                        help="Chebyshev radius for selecting nearby cubes")
    parser.add_argument("--rematch-every", type=int, default=200)
    parser.add_argument("--reheat-after", type=int, default=0,
                        help="reheat after this many trials without a new best (0 = off)")
    parser.add_argument("--reheat-t", type=float, default=None,
                        help="peak temperature of a reheat (default: sqrt(t0 * t1))")
    parser.add_argument("--no-fast-reject", action="store_true",
                        help="always call the simulator before the Metropolis decision (like v3)")
    parser.add_argument("--tag", default="v5", help="solution filename label")
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
    types_list = [int(t) for t in types.tolist()]
    target_positions = udp.target_cube_positions
    target_types = udp.target_cube_types
    n = len(initial_positions)
    num_types = setup["num_cube_types"]
    alpha = 0.1
    target_type_at = {
        tuple(position): int(cube_type)
        for position, cube_type in zip(target_positions.tolist(), target_types.tolist())
    }
    rematch = make_rematch(types, target_positions, target_types, num_types)
    p_mis, p_near, p_zero = args.p_mis, args.p_near, args.p_zero

    effects = learn_effects(initial_positions) if (args.p_dir > 0 or not args.no_fast_reject) else None
    fast = effects is not None and not args.no_fast_reject
    print("command steps:", effects if effects else "not fixed steps")
    print(f"fast reject: {'on' if fast else 'off'} | directed proposals: "
          f"{'on' if effects and args.p_dir > 0 else 'off'}")

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

    correct_flags = [target_type_at.get(tuple(p)) == types_list[i]
                     for i, p in enumerate(cubes.cube_position.tolist())]
    correct_count = sum(correct_flags)
    target_of = rematch(cubes.cube_position, np.array(correct_flags, dtype=bool))
    target_list = target_of.tolist()
    distances = np.abs(cubes.cube_position - target_of).sum(axis=1).tolist()
    total_distance = sum(distances)
    pool = [i for i in range(n) if distances[i] > 0]  # misplaced cubes
    pool_pos = [-1] * n
    for k, i in enumerate(pool):
        pool_pos[i] = k

    best_objective = (1 - alpha) * correct_count / n + alpha * (1 - len(moves) / max_commands)
    best_length, best_correct = len(moves), correct_count
    accepted = 0
    print(
        f"start: E={total_distance} | C={correct_count} | L={len(moves)} "
        f"| f={-((best_objective - setup['fitness_offset']) / (1 - setup['fitness_offset'])):.5f}"
    )

    def set_distance(cube_id, new_distance):
        nonlocal total_distance
        old = distances[cube_id]
        if old == new_distance:
            return
        total_distance += new_distance - old
        distances[cube_id] = new_distance
        if old == 0:
            pool_pos[cube_id] = len(pool)
            pool.append(cube_id)
        elif new_distance == 0:
            index = pool_pos[cube_id]
            last = pool[-1]
            pool[index] = last
            pool_pos[last] = index
            pool.pop()
            pool_pos[cube_id] = -1

    def do_rematch():
        nonlocal target_of, total_distance
        target_of = rematch(cubes.cube_position, np.array(correct_flags, dtype=bool))
        target_list[:] = target_of.tolist()
        distances[:] = np.abs(cubes.cube_position - target_of).sum(axis=1).tolist()
        total_distance = sum(distances)
        pool[:] = [i for i in range(n) if distances[i] > 0]
        pool_pos[:] = [-1] * n
        for k, i in enumerate(pool):
            pool_pos[i] = k

    def pick_cube():
        roll = rnd.random()
        if not pool or roll >= p_mis + p_near:
            return rnd.randrange(n)
        reference_cube = pool[rnd.randrange(len(pool))]
        if roll < p_mis:
            return reference_cube
        reference = (cubes.cube_position[reference_cube] if rnd.random() < 0.5
                     else target_of[reference_cube])
        nearby = np.flatnonzero(
            np.abs(cubes.cube_position - reference).max(axis=1) <= args.radius
        )
        return int(nearby[rnd.randrange(len(nearby))]) if len(nearby) else reference_cube

    def pick_command(pos, tgt, old_distance):
        if args.p_dir > 0 and effects is not None and old_distance > 0 and rnd.random() < args.p_dir:
            dx, dy, dz = tgt[0] - pos[0], tgt[1] - pos[1], tgt[2] - pos[2]
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
    apply_step = cubes.apply_single_update_step
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
        pos = cubes.cube_position[cube_id].tolist()
        tgt = target_list[cube_id]
        old_distance = distances[cube_id]
        command = pick_command(pos, tgt, old_distance)

        if fast:
            step = effects[command]
            new_distance = (abs(pos[0] + step[0] - tgt[0]) + abs(pos[1] + step[1] - tgt[1])
                            + abs(pos[2] + step[2] - tgt[2]))
            delta = new_distance - old_distance
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = rnd.random() < p_zero
            else:
                accept = rnd.random() < math.exp(-delta / temperature)
            if not accept:
                continue  # no simulator call at all
            if apply_step(cube_id, command) == 0:
                continue
            moves.append((cube_id, command))
        else:
            if apply_step(cube_id, command) == 0:
                continue
            new_pos = cubes.cube_position[cube_id].tolist()
            new_distance = (abs(new_pos[0] - tgt[0]) + abs(new_pos[1] - tgt[1])
                            + abs(new_pos[2] - tgt[2]))
            delta = new_distance - old_distance
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = rnd.random() < p_zero
            else:
                accept = rnd.random() < math.exp(-delta / temperature)
            if accept:
                moves.append((cube_id, command))
            else:
                undo_ok = apply_step(cube_id, command ^ 1)
                if undo_ok == 1 and cubes.cube_position[cube_id].tolist() == pos:
                    continue
                moves.append((cube_id, command))
                if undo_ok == 1:
                    moves.append((cube_id, command ^ 1))

        # bookkeeping from the real position (also correct if the step was not as predicted)
        new_pos = cubes.cube_position[cube_id].tolist()
        set_distance(cube_id, abs(new_pos[0] - tgt[0]) + abs(new_pos[1] - tgt[1])
                     + abs(new_pos[2] - tgt[2]))
        flag = target_type_at.get(tuple(new_pos)) == types_list[cube_id]
        if flag != correct_flags[cube_id]:
            correct_count += 1 if flag else -1
            correct_flags[cube_id] = flag

        current_objective = (1 - alpha) * correct_count / n + alpha * (1 - len(moves) / max_commands)
        if current_objective > best_objective:
            best_objective, best_length, best_correct = current_objective, len(moves), correct_count
            last_gain_trial = trial

        accepted += 1
        if accepted % args.rematch_every == 0:
            do_rematch()

    elapsed = time.time() - start_time

    # internal consistency check (cheap): tracked values against a recomputation
    positions_now = cubes.cube_position.tolist()
    recomputed = sum(abs(p[0] - t[0]) + abs(p[1] - t[1]) + abs(p[2] - t[2])
                     for p, t in zip(positions_now, target_list))
    recomputed_correct = sum(target_type_at.get(tuple(p)) == types_list[i]
                             for i, p in enumerate(positions_now))
    if recomputed != total_distance or recomputed_correct != correct_count \
            or len(pool) != sum(1 for d in distances if d > 0):
        print(f"WARNING: internal bookkeeping mismatch (E {total_distance} vs {recomputed}, "
              f"C {correct_count} vs {recomputed_correct})")

    moves = moves[:best_length]
    fitness, correct_final = official_result(udp, moves, types, target_positions, target_types, num_types)
    print(f"\n=== RESULT {args.tag} ({args.problem}) ===")
    print(
        f"f={fitness:.5f} | C={correct_final}/{n} | L={len(moves)} "
        f"| time={elapsed:.1f}s | seed={args.seed}"
    )
    if correct_final != best_correct:
        print(f"WARNING: tracked C={best_correct} differs from official C={correct_final}")
    if start_fitness is not None:
        print(f"start solution f={start_fitness:.5f}")

    os.makedirs("results", exist_ok=True)
    run_data = {
        "problem": args.problem,
        "fitness": fitness,
        "C": int(correct_final),
        "L": len(moves),
        "args": vars(args),
        "chromosome": [int(value) for move in moves for value in move] + [-1],
    }
    save_solution("annealing", args.problem, args.tag, args.seed, run_data)
    headers = ["problem", "tag", "seed", "t0", "t1", "trials", "p_mis", "p_near", "p_dir",
               "p_zero", "radius", "rematch_every", "fast", "start", "fitness", "C", "L", "seconds"]
    append_csv("results/annealing_v5_runs.csv", headers, [
        args.problem, args.tag, args.seed, args.t0, args.t1, args.trials, args.p_mis, args.p_near,
        args.p_dir, args.p_zero, args.radius, args.rematch_every, int(fast), args.start or "",
        round(fitness, 5), int(correct_final), len(moves), round(elapsed, 1),
    ])


if __name__ == "__main__":
    main()
