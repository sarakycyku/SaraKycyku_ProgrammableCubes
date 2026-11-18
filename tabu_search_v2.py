"""Tabu search v2 for the Programmable Cubes problems (ISS, JWST, Enterprise).

What changed compared with tabu_search.py (v1 hit the command budget after about 5000 iterations
on ISS, because every iteration appends one command, and every iteration spent up to 16
simulator calls):
  * Candidates are ranked ARITHMETICALLY (new distance and new position from the learned step of
    each command); the simulator is only called on the best-ranked candidates until one is legal,
    and that move is simply kept. Typically 1-3 simulator calls per iteration instead of 16.
    Falls back to the v1 apply/undo evaluation if the commands are not fixed steps (or --no-fast).
  * Rewind to the best solution: when the best objective has not improved for --patience
    iterations, or the command budget is nearly used, the state is rebuilt from the best prefix
    (the wasted commands are dropped), the tabu list is cleared and the search continues from
    there. This keeps L close to the best solution and makes all --iters iterations usable.
  * Exact set of misplaced cubes, cheaper rematching (as in annealing_v5.py).
The tabu list is keyed by (cube, position): a cube that just left a position may not return to it
for a random number of iterations in [tenure, 2*tenure]. Aspiration: a tabu move that puts a
misplaced cube exactly on its target is allowed.

Outputs: solutions/<problem>_tabu_<tag>_seed_<seed>.json, solutions/<problem>_tabu_<tag>.json,
results/tabu_v2_runs.csv. Use --start to continue from a saved solution.
Example:  python tabu_search_v2.py ISS --iters 300000 --seed 0 --tag v2
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes
from annealing_v5 import (learn_effects, make_rematch, official_result, append_csv,
                          save_solution)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem", nargs="?", default="ISS")
    parser.add_argument("--iters", type=int, default=300000, help="tabu iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=16,
                        help="candidate moves ranked per iteration")
    parser.add_argument("--tenure", type=int, default=30,
                        help="base tabu tenure; the real one is random in [tenure, 2*tenure]")
    parser.add_argument("--patience", type=int, default=5000,
                        help="iterations without a new best before rewinding to the best solution")
    parser.add_argument("--p-mis", type=float, default=0.7,
                        help="probability that a candidate cube is a misplaced one")
    parser.add_argument("--p-dir", type=float, default=0.5,
                        help="probability that a candidate command reduces the distance")
    parser.add_argument("--rematch-every", type=int, default=200)
    parser.add_argument("--start", default=None, help="JSON solution chromosome to resume from")
    parser.add_argument("--no-fast", action="store_true",
                        help="evaluate candidates with the simulator (apply/undo) like v1")
    parser.add_argument("--tag", default="v2")
    args = parser.parse_args()
    if args.iters <= 0 or args.seed < 0 or args.candidates <= 0 or args.tenure < 0 \
            or args.patience <= 0 or args.rematch_every <= 0:
        parser.error("--iters/--candidates/--patience/--rematch-every must be positive; "
                     "--seed/--tenure nonnegative")
    if not 0 <= args.p_mis <= 1 or not 0 <= args.p_dir <= 1:
        parser.error("--p-mis and --p-dir must be between 0 and 1")
    return args


class State:
    """Simulator state plus the bookkeeping that tabu search needs."""

    def __init__(self, initial_positions, types_list, target_type_at, rematch, max_commands):
        self.initial_positions = initial_positions
        self.types_list = types_list
        self.target_type_at = target_type_at
        self.rematch = rematch
        self.n = len(initial_positions)
        self.alpha = 0.1
        self.max_commands = max_commands
        self.cubes = None
        self.moves = []

    def load(self, moves):
        """Rebuild the state from scratch by replaying `moves`. Returns the number of illegal ones."""
        self.cubes = ProgrammableCubes(self.initial_positions)
        self.moves = []
        illegal = 0
        for cube_id, command in moves:
            self.moves.append((cube_id, command))
            if self.cubes.apply_single_update_step(cube_id, command) == 0:
                illegal += 1
        self.refresh()
        return illegal

    def refresh(self):
        positions = self.cubes.cube_position
        self.flags = [self.target_type_at.get(tuple(p)) == self.types_list[i]
                      for i, p in enumerate(positions.tolist())]
        self.correct = sum(self.flags)
        self.do_rematch()

    def do_rematch(self):
        positions = self.cubes.cube_position
        self.target_of = self.rematch(positions, np.array(self.flags, dtype=bool))
        self.target_list = self.target_of.tolist()
        self.distances = np.abs(positions - self.target_of).sum(axis=1).tolist()
        self.total = sum(self.distances)
        self.pool = [i for i in range(self.n) if self.distances[i] > 0]
        self.pool_pos = [-1] * self.n
        for k, i in enumerate(self.pool):
            self.pool_pos[i] = k

    def set_distance(self, cube_id, new_distance):
        old = self.distances[cube_id]
        if old == new_distance:
            return
        self.total += new_distance - old
        self.distances[cube_id] = new_distance
        if old == 0:
            self.pool_pos[cube_id] = len(self.pool)
            self.pool.append(cube_id)
        elif new_distance == 0:
            index = self.pool_pos[cube_id]
            last = self.pool[-1]
            self.pool[index] = last
            self.pool_pos[last] = index
            self.pool.pop()
            self.pool_pos[cube_id] = -1

    def update_cube(self, cube_id):
        """Bookkeeping of one cube from its real position after a move."""
        pos = self.cubes.cube_position[cube_id].tolist()
        tgt = self.target_list[cube_id]
        self.set_distance(cube_id, abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1]) + abs(pos[2] - tgt[2]))
        flag = self.target_type_at.get(tuple(pos)) == self.types_list[cube_id]
        if flag != self.flags[cube_id]:
            self.correct += 1 if flag else -1
            self.flags[cube_id] = flag

    def objective(self):
        return ((1 - self.alpha) * self.correct / self.n
                + self.alpha * (1 - len(self.moves) / self.max_commands))


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
    target_type_at = {tuple(p): int(t) for p, t in zip(target_positions.tolist(),
                                                       target_types.tolist())}
    rematch = make_rematch(types, target_positions, target_types, num_types)

    effects = learn_effects(initial_positions)
    fast = effects is not None and not args.no_fast
    print(f"command steps: {effects if effects else 'not fixed steps'} | fast candidates: "
          f"{'on' if fast else 'off'}")

    st = State(initial_positions, types_list, target_type_at, rematch, max_commands)
    start_moves = []
    if args.start:
        with open(args.start, encoding="utf-8") as source:
            chromosome = json.load(source)["chromosome"]
        for index in range(0, len(chromosome) - 1, 2):
            if chromosome[index] == -1:
                break
            start_moves.append((int(chromosome[index]), int(chromosome[index + 1])))
    illegal = st.load(start_moves)
    if args.start:
        print(f"start: loaded {len(start_moves)} commands | illegal: {illegal}")
    best_objective = st.objective()
    best_length, best_correct = len(st.moves), st.correct
    print(f"start: E={st.total} | C={st.correct} | L={len(st.moves)}")

    tabu = {}  # (cube, position tuple) -> iteration until which it is forbidden

    def pick_cube():
        if st.pool and rnd.random() < args.p_mis:
            return st.pool[rnd.randrange(len(st.pool))]
        return rnd.randrange(n)

    def pick_command(pos, tgt, old_distance):
        if effects is not None and old_distance > 0 and rnd.random() < args.p_dir:
            dx, dy, dz = tgt[0] - pos[0], tgt[1] - pos[1], tgt[2] - pos[2]
            base = abs(dx) + abs(dy) + abs(dz)
            good = [c for c, (ex, ey, ez) in enumerate(effects)
                    if abs(dx - ex) + abs(dy - ey) + abs(dz - ez) < base]
            if good:
                return good[rnd.randrange(len(good))]
        return rnd.randrange(6)

    def choose_fast(iteration):
        """Rank candidates by the predicted change; apply the best legal one. Returns (cube, old position)."""
        candidates = []
        for _ in range(args.candidates):
            cube_id = pick_cube()
            pos = st.cubes.cube_position[cube_id].tolist()
            tgt = st.target_list[cube_id]
            old_distance = st.distances[cube_id]
            command = pick_command(pos, tgt, old_distance)
            step = effects[command]
            new_pos = (pos[0] + step[0], pos[1] + step[1], pos[2] + step[2])
            new_distance = abs(new_pos[0] - tgt[0]) + abs(new_pos[1] - tgt[1]) + abs(new_pos[2] - tgt[2])
            if tabu.get((cube_id, new_pos), -1) > iteration and not (new_distance == 0 and old_distance > 0):
                continue
            candidates.append((new_distance - old_distance, rnd.random(), cube_id, command))
        candidates.sort()
        for _, _, cube_id, command in candidates:
            old_pos = tuple(st.cubes.cube_position[cube_id].tolist())
            if st.cubes.apply_single_update_step(cube_id, command) == 1:
                st.moves.append((cube_id, command))
                return cube_id, old_pos
        return None

    def choose_slow(iteration):
        """v1 behaviour: evaluate each candidate with apply/undo, keep the best non-tabu one."""
        best = None
        for _ in range(args.candidates):
            cube_id = pick_cube()
            pos = st.cubes.cube_position[cube_id].tolist()
            tgt = st.target_list[cube_id]
            old_distance = st.distances[cube_id]
            command = pick_command(pos, tgt, old_distance)
            if st.cubes.apply_single_update_step(cube_id, command) == 0:
                continue
            new_pos = tuple(st.cubes.cube_position[cube_id].tolist())
            new_distance = abs(new_pos[0] - tgt[0]) + abs(new_pos[1] - tgt[1]) + abs(new_pos[2] - tgt[2])
            undo_ok = st.cubes.apply_single_update_step(cube_id, command ^ 1)
            if not (undo_ok == 1 and st.cubes.cube_position[cube_id].tolist() == pos):
                # the probe could not be undone cleanly: it really happened, so keep it
                st.moves.append((cube_id, command))
                if undo_ok == 1:
                    st.moves.append((cube_id, command ^ 1))
                return cube_id, tuple(pos)
            if tabu.get((cube_id, new_pos), -1) > iteration and not (new_distance == 0 and old_distance > 0):
                continue
            key = (new_distance - old_distance, rnd.random(), cube_id, command)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        _, _, cube_id, command = best
        old_pos = tuple(st.cubes.cube_position[cube_id].tolist())
        if st.cubes.apply_single_update_step(cube_id, command) == 0:
            return None
        st.moves.append((cube_id, command))
        return cube_id, old_pos

    choose = choose_fast if fast else choose_slow
    start_time = time.time()
    report_every = max(1, args.iters // 20)
    applied = 0
    last_best_iteration = 0
    rewinds = 0
    for iteration in range(args.iters):
        if iteration % report_every == 0:
            print(f"it={iteration:7d} E={st.total:5d} C={st.correct:4d} L={len(st.moves):6d} "
                  f"| best: C={best_correct} L={best_length} | rewinds={rewinds} | tabu={len(tabu)}")
            for key in [k for k, v in tabu.items() if v <= iteration]:
                del tabu[key]
        out_of_budget = len(st.moves) >= max_commands - 2
        if out_of_budget or iteration - last_best_iteration >= args.patience:
            if best_length < len(st.moves):
                st.load(st.moves[:best_length])  # rewind to the best solution
                rewinds += 1
            elif out_of_budget:
                print("command budget reached at iteration", iteration)
                break
            tabu.clear()
            last_best_iteration = iteration
        picked = choose(iteration)
        if picked is None:
            continue
        cube_id, old_pos = picked
        tabu[(cube_id, old_pos)] = iteration + rnd.randint(args.tenure, 2 * args.tenure)
        st.update_cube(cube_id)
        value = st.objective()
        if value > best_objective:
            best_objective, best_length, best_correct = value, len(st.moves), st.correct
            last_best_iteration = iteration
        applied += 1
        if applied % args.rematch_every == 0:
            st.do_rematch()

    elapsed = time.time() - start_time
    moves = st.moves[:best_length]
    fitness, correct_final = official_result(udp, moves, types, target_positions, target_types, num_types)
    print(f"\n=== RESULT tabu {args.tag} ({args.problem}) ===")
    print(f"f={fitness:.5f} | C={correct_final}/{n} | L={len(moves)} | time={elapsed:.1f}s "
          f"| seed={args.seed} | rewinds={rewinds}")
    if correct_final != best_correct:
        print(f"WARNING: tracked C={best_correct} differs from official C={correct_final}")

    os.makedirs("results", exist_ok=True)
    run_data = {"problem": args.problem, "fitness": fitness, "C": int(correct_final),
                "L": len(moves), "args": vars(args),
                "chromosome": [int(v) for move in moves for v in move] + [-1]}
    save_solution("tabu", args.problem, args.tag, args.seed, run_data)
    headers = ["problem", "tag", "seed", "iters", "candidates", "tenure", "patience", "p_mis",
               "p_dir", "fast", "start", "fitness", "C", "L", "rewinds", "seconds"]
    append_csv("results/tabu_v2_runs.csv", headers, [
        args.problem, args.tag, args.seed, args.iters, args.candidates, args.tenure, args.patience,
        args.p_mis, args.p_dir, int(fast), args.start or "", round(fitness, 5), int(correct_final),
        len(moves), rewinds, round(elapsed, 1),
    ])


if __name__ == "__main__":
    main()
