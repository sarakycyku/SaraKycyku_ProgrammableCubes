"""fast_prune.py: remove redundant commands from a solution, using the fast exact simulator.

Same idea as prune.py: if a cube leaves a position and later returns to exactly that position,
all commands of that cube in between (an "excursion") may be useless. A removal is accepted only
if the whole remaining command list is still legal and the number of correct cubes C does not
drop. Groups of excursions are tested together and split in halves when the test fails (group
testing). Here a full replay takes milliseconds (numba), so no checkpoint tricks are needed and the
rules are exactly the official ones.

Usage:
  python fast_prune.py JWST solutions/jwst_fastsa_base_seed_0.json --max-minutes 10 --out solutions/jwst_pruned_f.json
The result is verified with the official simulator and saved only if it is better than --out.
"""
import argparse
import json
import os
import time

import numpy as np
from numba import njit

from fastcubes import FastCubes, apply_move, DX, DY, DZ
from fast_sa import Context, load_moves
from annealing_v5 import official_result


@njit(cache=True)
def replay_keep(grid0, pos0, cubes_arr, cmds_arr, keep, count, tgt_type_grid, types,
                offs, occ_mask, emp_mask, opts, disp, dx, dy, dz):
    """Replay the commands with keep[i] True. Returns the number of correct cubes, or -1 if any
    kept command is illegal."""
    grid = grid0.copy()
    pos = pos0.copy()
    n = pos.shape[0]
    queue = np.empty(n + 6, dtype=np.int64)
    visit = np.zeros(n, dtype=np.int64)
    stampbox = np.zeros(1, dtype=np.int64)
    nbbuf = np.empty(6, dtype=np.int64)
    for i in range(count):
        if keep[i]:
            if apply_move(grid, pos, cubes_arr[i], cmds_arr[i], offs, occ_mask, emp_mask, opts, disp,
                          queue, visit, stampbox, nbbuf, dx, dy, dz) == 0:
                return -1
    correct = 0
    for c in range(n):
        if tgt_type_grid[pos[c, 0], pos[c, 1], pos[c, 2]] == types[c]:
            correct += 1
    return correct


@njit(cache=True)
def record_positions(grid0, pos0, cubes_arr, cmds_arr, count, offs, occ_mask, emp_mask, opts, disp,
                     dx, dy, dz, before, after):
    """Replay everything and store the flat position of the moved cube before and after each command.
    Returns the number of illegal commands (0 for a valid solution)."""
    grid = grid0.copy()
    pos = pos0.copy()
    n = pos.shape[0]
    sy = grid.shape[1]
    sz = grid.shape[2]
    queue = np.empty(n + 6, dtype=np.int64)
    visit = np.zeros(n, dtype=np.int64)
    stampbox = np.zeros(1, dtype=np.int64)
    nbbuf = np.empty(6, dtype=np.int64)
    illegal = 0
    for i in range(count):
        c = cubes_arr[i]
        before[i] = (pos[c, 0] * sy + pos[c, 1]) * sz + pos[c, 2]
        if apply_move(grid, pos, c, cmds_arr[i], offs, occ_mask, emp_mask, opts, disp,
                      queue, visit, stampbox, nbbuf, dx, dy, dz) == 0:
            illegal += 1
        after[i] = (pos[c, 0] * sy + pos[c, 1]) * sz + pos[c, 2]
    return illegal


class Pruner:
    def __init__(self, ctx):
        self.ctx = ctx
        base = FastCubes(ctx.init, extra_positions=ctx.target_positions)
        self.grid0, self.pos0 = base.grid, base.pos
        self.tables = base.tables
        self.replays = 0

    def arrays(self, moves):
        arr = np.asarray(moves, dtype=np.int64).reshape(-1, 2)
        return np.ascontiguousarray(arr[:, 0]), np.ascontiguousarray(arr[:, 1])

    def count_correct(self, cubes_arr, cmds_arr, keep):
        occ_mask, emp_mask, opts, offs, disp = self.tables
        self.replays += 1
        return replay_keep(self.grid0, self.pos0, cubes_arr, cmds_arr, keep, len(cubes_arr),
                           self.ctx.tgt_type_grid, self.ctx.types, offs, occ_mask, emp_mask, opts,
                           disp, DX, DY, DZ)

    def excursions(self, cubes_arr, cmds_arr):
        occ_mask, emp_mask, opts, offs, disp = self.tables
        count = len(cubes_arr)
        before = np.zeros(count, dtype=np.int64)
        after = np.zeros(count, dtype=np.int64)
        illegal = record_positions(self.grid0, self.pos0, cubes_arr, cmds_arr, count, offs, occ_mask,
                                   emp_mask, opts, disp, DX, DY, DZ, before, after)
        if illegal:
            raise SystemExit(f"The input solution has {illegal} illegal commands; cannot prune safely.")
        by_cube = {}
        for t in range(count):
            by_cube.setdefault(int(cubes_arr[t]), []).append(t)
        groups = []
        for c, idx in by_cube.items():
            seen = {int(before[idx[0]]): -1}
            for k, t in enumerate(idx):
                a = int(after[t])
                if a in seen:
                    groups.append(idx[seen[a] + 1:k + 1])
                    seen = {a: k}
                else:
                    seen[a] = k
        return groups

    def prune(self, moves, deadline, verbose=True):
        cubes_arr, cmds_arr = self.arrays(moves)
        keep = np.ones(len(moves), dtype=np.bool_)
        c0 = self.count_correct(cubes_arr, cmds_arr, keep)
        if c0 < 0:
            raise SystemExit("The input solution has illegal commands; cannot prune safely.")
        rounds = 0
        started = time.time()
        while time.time() < deadline:
            groups = self.excursions(cubes_arr, cmds_arr)
            if not groups:
                break
            groups.sort(key=len)
            accepted = np.zeros(len(moves), dtype=np.bool_)
            stack = [groups]
            removed_total = 0
            while stack and time.time() < deadline:
                gs = stack.pop()
                trial = accepted.copy()
                for g in gs:
                    trial[g] = True
                if self.count_correct(cubes_arr, cmds_arr, ~trial) >= c0:
                    accepted = trial
                elif len(gs) > 1:
                    mid = len(gs) // 2
                    stack.append(gs[mid:])
                    stack.append(gs[:mid])
            removed_total = int(accepted.sum())
            if removed_total == 0:
                break
            moves = [m for m, rm in zip(moves, accepted) if not rm]
            cubes_arr, cmds_arr = self.arrays(moves)
            rounds += 1
            if verbose:
                print(f"round {rounds}: removed {removed_total} -> L = {len(moves)} "
                      f"({len(groups)} groups, {self.replays} replays, {time.time() - started:.1f} s)",
                      flush=True)
        return moves


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem")
    parser.add_argument("solution")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-minutes", type=float, default=10.0)
    args = parser.parse_args()
    ctx = Context(args.problem)
    moves = load_moves(args.solution)
    f0, c0 = official_result(ctx.udp, moves, ctx.udp.initial_cube_types, ctx.target_positions,
                             ctx.udp.target_cube_types, ctx.num_types)
    print(f"input: L={len(moves)} C={c0} f={f0:.5f}")
    started = time.time()
    pruned = Pruner(ctx).prune(moves, started + args.max_minutes * 60)
    f, c = official_result(ctx.udp, pruned, ctx.udp.initial_cube_types, ctx.target_positions,
                           ctx.udp.target_cube_types, ctx.num_types)
    print(f"\n=== PRUNED ({args.problem}) ===\nL: {len(moves)} -> {len(pruned)} | C: {c0} -> {c} | "
          f"f: {f0:.5f} -> {f:.5f} | {time.time() - started:.1f} s")
    out = args.out or f"solutions/{args.problem.lower()}_pruned_fast.json"
    os.makedirs("solutions", exist_ok=True)
    better = True
    if os.path.exists(out):
        with open(out, encoding="utf-8") as handle:
            better = f < json.load(handle)["fitness"]
    if c < c0:
        print("WARNING: official C dropped; not saved.")
    elif better:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"problem": args.problem, "fitness": f, "C": int(c), "L": len(pruned),
                       "source": args.solution,
                       "chromosome": [int(v) for mv in pruned for v in mv] + [-1]}, handle)
        print("saved", out)
    else:
        print("not saved (not better than the stored file)")


if __name__ == "__main__":
    main()
