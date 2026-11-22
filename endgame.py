"""endgame.py: finish a solution by exact search, one cube at a time.

Simulated annealing guided by L1 distance gets stuck with a few cubes that are only 1-3 cells from
a free target cell: no single pivot reduces the distance, but a longer pivot sequence reaches it.
This script takes a solution, and for every cube that is not on a target cell of its own type it
runs a breadth-first search over the positions that cube can reach by legal pivots (all other
cubes fixed, exact rules from fastcubes.py) and stops at the nearest FREE target cell of the cube's
type. The shortest path is appended to the solution, which raises C by one each time. It repeats
until no cube can improve any more, or the command budget is used.

Usage:  python endgame.py ISS --start solutions/iss_fastsa_f1.json --tag e1
Outputs: solutions/<problem>_endgame_<tag>.json (always verified with the official simulator).
"""
import argparse
import json
import os
import time

import numpy as np
from numba import njit

from fastcubes import FastCubes, peek_move, stays_connected, DX, DY, DZ
from fast_sa import Context, load_moves
from annealing_v5 import official_result, save_solution, append_csv


@njit(cache=True)
def bfs_cube(grid, pos, c, ctype, tgt_type_grid, seen, stamp, parent, pcmd, queue,
             cq, visit, stampbox, nbbuf, offs, occ_mask, emp_mask, opts, disp, dx, dy, dz,
             max_nodes, out_cmds):
    """Shortest legal pivot path of cube c (others fixed) to a free target cell of type ctype.
    Returns the path length (0 if none); commands are written to out_cmds in order."""
    sx, sy, sz = grid.shape
    x0 = pos[c, 0]
    y0 = pos[c, 1]
    z0 = pos[c, 2]
    start = (x0 * sy + y0) * sz + z0
    grid[x0, y0, z0] = -1
    stamp[0] += 1
    st = stamp[0]
    seen[start] = st
    parent[start] = -1
    queue[0] = start
    qh = 0
    qt = 1
    goal = -1
    x = x0
    y = y0
    z = z0
    while qh < qt and goal < 0 and qt < max_nodes:
        cur = queue[qh]
        qh += 1
        x = cur // (sy * sz)
        y = (cur // sz) % sy
        z = cur % sz
        pos[c, 0] = x
        pos[c, 1] = y
        pos[c, 2] = z
        grid[x, y, z] = c
        for cmd in range(6):
            ok, nx, ny, nz = peek_move(grid, pos, c, cmd, offs, occ_mask, emp_mask, opts, disp)
            if not ok:
                continue
            q = (nx * sy + ny) * sz + nz
            if seen[q] == st:
                continue
            if not stays_connected(grid, pos, c, cq, visit, stampbox, nbbuf, dx, dy, dz):
                continue
            seen[q] = st
            parent[q] = cur
            pcmd[q] = cmd
            queue[qt] = q
            qt += 1
            if tgt_type_grid[nx, ny, nz] == ctype and grid[nx, ny, nz] < 0:
                goal = q
                break
        grid[x, y, z] = -1
    pos[c, 0] = x0
    pos[c, 1] = y0
    pos[c, 2] = z0
    grid[x0, y0, z0] = c
    if goal < 0:
        return 0
    length = 0
    node = goal
    while parent[node] >= 0:
        length += 1
        node = parent[node]
    node = goal
    i = length - 1
    while parent[node] >= 0:
        out_cmds[i] = pcmd[node]
        i -= 1
        node = parent[node]
    return length


class Endgame:
    """Exact single-cube repair (BFS) plus kick-and-repair local search on top of it."""

    def __init__(self, ctx, moves, max_nodes=2000000, seed=0):
        self.ctx = ctx
        self.sim = FastCubes(ctx.init, extra_positions=ctx.target_positions)
        self.illegal = self.sim.apply_moves(moves) if moves else 0
        self.moves = list(moves)
        self.rng = np.random.default_rng(seed)
        self.max_nodes = max_nodes
        size = self.sim.grid.size
        self.seen = np.zeros(size, dtype=np.int64)
        self.stamp = np.zeros(1, dtype=np.int64)
        self.parent = np.zeros(size, dtype=np.int64)
        self.pcmd = np.zeros(size, dtype=np.int64)
        self.queue = np.zeros(size, dtype=np.int64)
        self.out_cmds = np.zeros(size, dtype=np.int64)
        self.budget = ctx.max_cmds - 2

    def flags(self):
        p = self.sim.pos
        return self.ctx.tgt_type_grid[p[:, 0], p[:, 1], p[:, 2]] == self.ctx.types

    def correct(self):
        return int(self.flags().sum())

    def bfs(self, c):
        sim, ctx = self.sim, self.ctx
        occ_mask, emp_mask, opts, offs, disp = sim.tables
        return bfs_cube(sim.grid, sim.pos, int(c), int(ctx.types[c]), ctx.tgt_type_grid, self.seen,
                        self.stamp, self.parent, self.pcmd, self.queue, sim.queue, sim.visit,
                        sim.stampbox, sim.nbbuf, offs, occ_mask, emp_mask, opts, disp, DX, DY, DZ,
                        self.max_nodes, self.out_cmds)

    def repair(self):
        """Move wrong cubes to free target cells of their type, shortest paths, until none can."""
        improved_any = True
        while improved_any:
            improved_any = False
            wrong = np.flatnonzero(~self.flags())
            self.rng.shuffle(wrong)
            for c in wrong:
                if self.flags()[c]:
                    continue
                length = self.bfs(c)
                if length == 0 or len(self.moves) + length > self.budget:
                    continue
                for i in range(length):
                    assert self.sim.apply(int(c), int(self.out_cmds[i])) == 1, "BFS path must be legal"
                    self.moves.append((int(c), int(self.out_cmds[i])))
                improved_any = True
        return self.correct()

    def snapshot(self):
        return self.sim.pos.copy(), self.sim.grid.copy(), len(self.moves)

    def restore(self, snap):
        pos, grid, length = snap
        self.sim.pos[:] = pos
        self.sim.grid[:] = grid
        del self.moves[length:]

    def kick_and_repair(self, iterations, depth=2, radius=3, verbose=True, time_limit=None):
        """Random helper moves near wrong cubes, then exact repair; keep only if C increases."""
        started = time.time()
        best_c = self.correct()
        accepted = 0
        for it in range(iterations):
            if time_limit and time.time() - started > time_limit:
                break
            wrong = np.flatnonzero(~self.flags())
            if len(wrong) == 0:
                break
            snap = self.snapshot()
            w = int(self.rng.choice(wrong))
            center = self.sim.pos[w]
            near = np.flatnonzero(np.abs(self.sim.pos - center).max(axis=1) <= radius)
            applied = 0
            for _ in range(depth):
                h = int(self.rng.choice(near))
                cmd = int(self.rng.integers(6))
                if self.sim.apply(h, cmd) == 1:
                    self.moves.append((h, cmd))
                    applied += 1
            if applied == 0:
                continue
            c_now = self.repair()
            if c_now > best_c and len(self.moves) <= self.budget:
                best_c = c_now
                accepted += 1
                if verbose:
                    print(f"  kick {it}: C -> {c_now} | L={len(self.moves)} ({time.time() - started:.1f} s)", flush=True)
            else:
                self.restore(snap)
        return best_c, accepted


def complete(ctx, moves, max_nodes=2000000, seed=0, verbose=True, budget=None):
    eg = Endgame(ctx, moves, max_nodes, seed)
    if budget:
        eg.budget = budget
    c = eg.repair()
    return eg.moves, c


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem")
    parser.add_argument("--start", required=True, help="solution JSON to finish")
    parser.add_argument("--tag", default="e1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kicks", type=int, default=0, help="kick-and-repair iterations (0 = only BFS repair)")
    parser.add_argument("--depth", type=int, default=2, help="helper moves per kick")
    parser.add_argument("--radius", type=int, default=3, help="helper cubes within this Chebyshev distance of a wrong cube")
    parser.add_argument("--time-limit", type=float, default=None, help="seconds for the kick phase")
    args = parser.parse_args()
    ctx = Context(args.problem)
    moves = load_moves(args.start)
    f0, c0 = official_result(ctx.udp, moves, ctx.udp.initial_cube_types, ctx.target_positions,
                             ctx.udp.target_cube_types, ctx.num_types)
    print(f"input: f={f0:.5f} C={c0}/{ctx.n} L={len(moves)}")
    started = time.time()
    eg = Endgame(ctx, moves, seed=args.seed)
    c_tracked = eg.repair()
    print(f"after BFS repair: C={c_tracked} L={len(eg.moves)} ({time.time() - started:.1f} s)")
    if args.kicks:
        c_tracked, accepted = eg.kick_and_repair(args.kicks, args.depth, args.radius, time_limit=args.time_limit)
        print(f"kick phase: {accepted} accepted kicks")
    new_moves = eg.moves
    f, c = official_result(ctx.udp, new_moves, ctx.udp.initial_cube_types, ctx.target_positions,
                           ctx.udp.target_cube_types, ctx.num_types)
    print(f"\n=== ENDGAME ({args.problem}) ===\nf: {f0:.5f} -> {f:.5f} | C: {c0} -> {c} | "
          f"L: {len(moves)} -> {len(new_moves)} | {time.time() - started:.1f} s")
    if c != c_tracked:
        print(f"WARNING: tracked C={c_tracked} differs from official C={c}")
    run_data = {"problem": args.problem, "fitness": f, "C": int(c), "L": len(new_moves),
                "source": args.start, "chromosome": [int(v) for mv in new_moves for v in mv] + [-1]}
    save_solution("endgame", args.problem, args.tag, args.seed, run_data)


if __name__ == "__main__":
    main()
