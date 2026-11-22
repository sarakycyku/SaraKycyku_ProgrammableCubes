"""fast_sa.py: simulated annealing running entirely on the fast exact simulator (fastcubes.py).

Same search as annealing_v3/v5 (random cube, random command, Metropolis on the L1 distance to the
matched target, periodic optimal rematching), but every trial is a few array reads inside one
numba loop:
  * PEEK first: where would the command move the cube? (pattern check only)
  * Metropolis decision on the exact new distance;
  * only an accepted move pays for the connectivity check and is applied.
There is no undo, so rejected proposals never add commands (v3/v5 had to record an undo pair
whenever the inverse command did not restore the position).

Schedule: progress = max(trials done / --trials, commands used / (cap_frac * max_cmds)); the
temperature falls from --t0 to --t1 with progress and the run ends when progress reaches 1. So
the search is cooled by the command budget (the real constraint), and --trials is only an upper
bound. A colder setting (lower t0/t1/p-zero) makes the search spend more trials per command.
The final solution is always re-evaluated with the OFFICIAL simulator (udp.fitness).

Outputs: solutions/<problem>_fastsa_<tag>_seed_<seed>.json, solutions/<problem>_fastsa_<tag>.json
(best over seeds), results/fast_sa_runs.csv. Use --start to continue from a saved solution (any
solution file with a "chromosome" works, also those of annealing_v3/v4/v5 and the pruned ones).
"""
import argparse
import json
import math
import os
import time

import numpy as np
from numba import njit

from programmable_cubes_UDP import programmable_cubes_UDP
from fastcubes import FastCubes, peek_move, stays_connected, commit_move, DX, DY, DZ
from annealing_v5 import make_rematch, official_result, append_csv, save_solution

ALPHA = 0.1


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def sa_chunk(grid, pos, tgt, dist, flags, types, tgt_type_grid, pool, pool_pos, st, bestobj,
             mv_cube, mv_cmd, queue, visit, stampbox, nbbuf,
             offs, occ_mask, emp_mask, opts, disp, dx, dy, dz,
             k0, total, t0, t1, p_mis, p_zero, rematch_every, max_cmds, alpha, lcap):
    """Run trials from k0 until `rematch_every` moves were accepted, the budget is used or the
    schedule ends. st = [correct, total_distance, nmoves, best_len, best_correct, pool_size,
    accepted_since_rematch, finished]. Returns the index of the next trial."""
    n = pos.shape[0]
    log_ratio = math.log(t1 / t0)
    k = k0
    while k < total:
        if st[2] >= max_cmds - 2 or st[6] >= rematch_every:
            break
        progress = max(k / total, st[2] / lcap)
        if progress >= 1.0:
            st[7] = 1
            break
        T = t0 * math.exp(log_ratio * progress)
        k += 1
        if st[5] > 0 and np.random.random() < p_mis:
            c = pool[int(np.random.random() * st[5])]
        else:
            c = np.random.randint(0, n)
        cmd = np.random.randint(0, 6)
        ok, nx, ny, nz = peek_move(grid, pos, c, cmd, offs, occ_mask, emp_mask, opts, disp)
        if not ok:
            continue
        old_d = dist[c]
        new_d = abs(nx - tgt[c, 0]) + abs(ny - tgt[c, 1]) + abs(nz - tgt[c, 2])
        delta = new_d - old_d
        if delta < 0:
            acc = True
        elif delta == 0:
            acc = np.random.random() < p_zero
        else:
            acc = np.random.random() < math.exp(-delta / T)
        if not acc:
            continue
        if not stays_connected(grid, pos, c, queue, visit, stampbox, nbbuf, dx, dy, dz):
            continue
        commit_move(grid, pos, c, nx, ny, nz)
        mv_cube[st[2]] = c
        mv_cmd[st[2]] = cmd
        st[2] += 1
        st[1] += new_d - old_d
        dist[c] = new_d
        if old_d == 0:
            pool_pos[c] = st[5]
            pool[st[5]] = c
            st[5] += 1
        elif new_d == 0:
            i = pool_pos[c]
            last = pool[st[5] - 1]
            pool[i] = last
            pool_pos[last] = i
            st[5] -= 1
            pool_pos[c] = -1
        flag = 1 if tgt_type_grid[nx, ny, nz] == types[c] else 0
        if flag != flags[c]:
            st[0] += 1 if flag == 1 else -1
            flags[c] = flag
        obj = (1.0 - alpha) * st[0] / n + alpha * (1.0 - st[2] / max_cmds)
        if obj > bestobj[0]:
            bestobj[0] = obj
            st[3] = st[2]
            st[4] = st[0]
        st[6] += 1
    return k


class Context:
    """Problem data shared by many runs (so a tuning loop does not reload anything)."""

    def __init__(self, problem):
        self.problem = problem
        self.udp = programmable_cubes_UDP(problem)
        setup = self.udp.setup
        self.max_cmds = setup["max_cmds"]
        self.offset = setup["fitness_offset"]
        self.num_types = setup["num_cube_types"]
        self.init = np.load(f"{setup['path']}/Initial_Config.npy")
        self.types = self.udp.initial_cube_types.astype(np.int64)
        self.target_positions = self.udp.target_cube_positions
        self.target_types = self.udp.target_cube_types.astype(np.int64)
        self.n = len(self.init)
        probe = FastCubes(self.init, extra_positions=self.target_positions)
        self.origin = probe.origin
        self.grid_shape = probe.grid.shape
        self.target_g = (self.target_positions - self.origin).astype(np.int64)
        self.tgt_type_grid = np.full(self.grid_shape, -1, dtype=np.int64)
        self.tgt_type_grid[self.target_g[:, 0], self.target_g[:, 1], self.target_g[:, 2]] = self.target_types
        self.rematch = make_rematch(self.types, self.target_g, self.target_types, self.num_types)

    def f_of(self, correct, length):
        score = (1 - ALPHA) * correct / self.n + ALPHA * (1 - length / self.max_cmds)
        return -(score - self.offset) / (1 - self.offset)


def anneal(ctx, seed, trials, t0, t1, p_mis=0.5, p_zero=0.3, cap_frac=1.0, rematch_every=200,
           start_moves=None, verbose=False):
    """One annealing run. Returns dict(f, C, L, cube=array, cmd=array, seconds, trials_done)."""
    n, max_cmds = ctx.n, ctx.max_cmds
    sim = FastCubes(ctx.init, extra_positions=ctx.target_positions)
    mv_cube = np.zeros(max_cmds + 8, dtype=np.int64)
    mv_cmd = np.zeros(max_cmds + 8, dtype=np.int64)
    nmoves = 0
    illegal = 0
    if start_moves:
        illegal = sim.apply_moves(start_moves)
        for i, (c, m) in enumerate(start_moves):
            mv_cube[i], mv_cmd[i] = c, m
        nmoves = len(start_moves)
    pos = sim.pos
    flags = (ctx.tgt_type_grid[pos[:, 0], pos[:, 1], pos[:, 2]] == ctx.types).astype(np.int64)
    tgt = np.zeros_like(pos)
    dist = np.zeros(n, dtype=np.int64)
    pool = np.zeros(n, dtype=np.int64)
    pool_pos = np.full(n, -1, dtype=np.int64)
    st = np.zeros(8, dtype=np.int64)
    bestobj = np.zeros(1)

    def refresh():
        tgt[:] = ctx.rematch(pos, flags.astype(bool))
        dist[:] = np.abs(pos - tgt).sum(axis=1)
        mis = np.flatnonzero(dist > 0)
        pool[:len(mis)] = mis
        pool_pos[:] = -1
        pool_pos[mis] = np.arange(len(mis))
        st[1] = dist.sum()
        st[5] = len(mis)

    refresh()
    st[0] = flags.sum()
    st[2] = nmoves
    st[3], st[4] = nmoves, st[0]
    bestobj[0] = (1 - ALPHA) * st[0] / n + ALPHA * (1 - nmoves / max_cmds)
    if verbose:
        print(f"start: E={st[1]} C={st[0]} L={nmoves} | illegal in start list: {illegal}")
    seed_rng(seed)
    occ_mask, emp_mask, opts, offs, disp = sim.tables
    lcap = max(1.0, cap_frac * (max_cmds - 2))
    report_every = max(1, trials // 20)
    next_report = 0
    k = 0
    started = time.time()
    while k < trials and st[7] == 0:
        if verbose and k >= next_report:
            prog = max(k / trials, st[2] / lcap)
            print(f"k={k:10d} T={t0 * (t1 / t0) ** min(1.0, prog):.3f} E={st[1]:5d} C={st[0]:4d} "
                  f"L={st[2]:6d} | best: C={st[4]} L={st[3]} | {time.time() - started:.0f}s", flush=True)
            next_report += report_every
        if st[2] >= max_cmds - 2:
            break
        st[6] = 0
        k = sa_chunk(sim.grid, sim.pos, tgt, dist, flags, ctx.types, ctx.tgt_type_grid, pool,
                     pool_pos, st, bestobj, mv_cube, mv_cmd, sim.queue, sim.visit, sim.stampbox,
                     sim.nbbuf, offs, occ_mask, emp_mask, opts, disp, DX, DY, DZ,
                     k, trials, t0, t1, p_mis, p_zero, rematch_every, max_cmds, ALPHA, lcap)
        refresh()
    best_len = int(st[3])
    return {"f": ctx.f_of(int(st[4]), best_len), "C": int(st[4]), "L": best_len,
            "cube": mv_cube[:best_len].copy(), "cmd": mv_cmd[:best_len].copy(),
            "seconds": time.time() - started, "trials_done": int(k)}


def load_moves(path):
    with open(path, encoding="utf-8") as source:
        chrom = json.load(source)["chromosome"]
    return [(int(chrom[i]), int(chrom[i + 1])) for i in range(0, len(chrom) - 1, 2) if chrom[i] != -1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem", nargs="?", default="ISS")
    parser.add_argument("--trials", type=int, default=50000000, help="upper bound on trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t0", type=float, default=0.5)
    parser.add_argument("--t1", type=float, default=0.25)
    parser.add_argument("--cap-frac", type=float, default=1.0)
    parser.add_argument("--p-mis", type=float, default=0.5)
    parser.add_argument("--p-zero", type=float, default=0.3)
    parser.add_argument("--rematch-every", type=int, default=200)
    parser.add_argument("--start", default=None)
    parser.add_argument("--tag", default="f1")
    args = parser.parse_args()
    if args.trials <= 0 or args.t0 <= 0 or args.t1 <= 0 or args.rematch_every <= 0 or args.cap_frac <= 0:
        parser.error("--trials, --t0, --t1, --cap-frac and --rematch-every must be positive")
    return args


def main():
    args = parse_args()
    ctx = Context(args.problem)
    start_moves = load_moves(args.start) if args.start else None
    result = anneal(ctx, args.seed, args.trials, args.t0, args.t1, args.p_mis, args.p_zero,
                    args.cap_frac, args.rematch_every, start_moves, verbose=True)
    moves = list(zip(result["cube"].tolist(), result["cmd"].tolist()))
    fitness, correct_final = official_result(ctx.udp, moves, ctx.udp.initial_cube_types,
                                             ctx.target_positions, ctx.udp.target_cube_types, ctx.num_types)
    print(f"\n=== RESULT fastsa {args.tag} ({args.problem}) ===")
    print(f"f={fitness:.5f} | C={correct_final}/{ctx.n} | L={len(moves)} | time={result['seconds']:.1f}s "
          f"| seed={args.seed} | trials={result['trials_done']}")
    if correct_final != result["C"]:
        print(f"WARNING: tracked C={result['C']} differs from official C={correct_final}")
    os.makedirs("results", exist_ok=True)
    run_data = {"problem": args.problem, "fitness": fitness, "C": int(correct_final),
                "L": len(moves), "args": vars(args),
                "chromosome": [int(v) for move in moves for v in move] + [-1]}
    save_solution("fastsa", args.problem, args.tag, args.seed, run_data)
    headers = ["problem", "tag", "seed", "t0", "t1", "trials", "p_mis", "p_zero", "cap_frac",
               "rematch_every", "start", "fitness", "C", "L", "seconds", "trials_done"]
    append_csv("results/fast_sa_runs.csv", headers, [
        args.problem, args.tag, args.seed, args.t0, args.t1, args.trials, args.p_mis, args.p_zero,
        args.cap_frac, args.rematch_every, args.start or "", round(fitness, 5), int(correct_final),
        len(moves), round(result["seconds"], 1), result["trials_done"]])


if __name__ == "__main__":
    main()
