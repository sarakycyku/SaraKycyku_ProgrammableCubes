# prune_fast.py: faster version of prune.py (same idea, much cheaper tests).
#
# prune.py removes "excursions" (a cube leaves a position and later returns to exactly that
# position) and checks each attempt by replaying the WHOLE command list from the start.
# Here a test only replays the part of the list that can change:
#   * the replay starts at a stored checkpoint just before the first removed command
#     (the simulator is rebuilt from the checkpoint positions);
#   * it stops as soon as the state is identical to the original state at a later checkpoint
#     (then the rest of the list is identical, so it is legal and gives the same result).
# Groups of excursions are tested in windows ordered by position, with group testing (halving)
# on failure, exactly like prune.py. A removal is accepted only if it converges back to the
# original state (this covers the excursion case; removals that leave other cubes in different
# places are not accepted, unlike prune.py).
#
# Safety: at start-up a self-test checks that rebuilding the simulator from checkpoint positions
# gives the same behaviour and the same internal state as continuing the original simulation.
# If it does not, the script stops and says so (use prune.py then). The result is always
# verified with the official simulator and is saved only if C does not drop and f improves.
#
# Usage (from the folder with programmable_cubes_UDP.py):
#   python prune_fast.py JWST solutions/jwst_annealing_v5.json --max-minutes 20 --out solutions/jwst_sa_pruned.json
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes

ap = argparse.ArgumentParser()
ap.add_argument("problem")
ap.add_argument("solution", help="JSON with the chromosome")
ap.add_argument("--out", default=None, help="output file (default solutions/<problem>_pruned.json)")
ap.add_argument("--max-minutes", type=float, default=20.0)
ap.add_argument("--checkpoint-every", type=int, default=256, help="K: moves between checkpoints")
ap.add_argument("--max-groups", type=int, default=64, help="groups per testing window")
ap.add_argument("--span", type=int, default=4096, help="maximum window length in moves")
ap.add_argument("--max-tail", type=int, default=20,
                help="give up a test that has not converged after this many checkpoints")
args = ap.parse_args()
K = args.checkpoint_every

udp = programmable_cubes_UDP(args.problem)
s = udp.setup
init_pos = np.load("{}/Initial_Config.npy".format(s["path"]))
types = udp.initial_cube_types
tgt_pos = udp.target_cube_positions
tgt_types = udp.target_cube_types
n = len(init_pos)
target_type_at = {tuple(p): int(t) for p, t in zip(tgt_pos.tolist(), tgt_types.tolist())}
types_list = [int(t) for t in types.tolist()]

with open(args.solution, encoding="utf-8") as fh:
    data = json.load(fh)
chrom = data["chromosome"]
moves = []
for i in range(0, len(chrom) - 1, 2):
    if chrom[i] == -1:
        break
    moves.append((int(chrom[i]), int(chrom[i + 1])))


def count_correct(positions):
    return sum(1 for i, p in enumerate(positions.tolist())
               if target_type_at.get(tuple(p)) == types_list[i])


def same_attr(a, b):
    """True/False if comparable, None if not."""
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(a, b))
        return bool(a == b)
    except Exception:
        return None


class Record:
    """Original replay: checkpoints every K moves, final positions, per-move history."""

    def __init__(self, mv, keep_object_at=None):
        cubes = ProgrammableCubes(init_pos)
        self.L = len(mv)
        self.states = []          # positions before move t, for t = 0, K, 2K, ...
        self.hist = []            # (before, after) position tuples of the moved cube
        self.ok = True
        self.snapshot = None
        for t, (c, m) in enumerate(mv):
            if t % K == 0:
                self.states.append(cubes.cube_position.copy())
                if t == keep_object_at:
                    try:
                        self.snapshot = {k: copy.deepcopy(v) for k, v in vars(cubes).items()}
                    except Exception:
                        self.snapshot = None
            before = tuple(cubes.cube_position[c].tolist())
            if cubes.apply_single_update_step(c, m) == 0:
                self.ok = False
                return
            self.hist.append((before, tuple(cubes.cube_position[c].tolist())))
        self.final = cubes.cube_position.copy()

    def state_at(self, t):
        return self.final if t == self.L else self.states[t // K]


def self_test(mv, rec):
    """Rebuilding the simulator from checkpoint positions must behave like the original run."""
    if len(mv) < 3 * K:
        return True, "list too short for a meaningful self-test (skipped)"
    c = (len(mv) // (2 * K)) * K
    fresh = ProgrammableCubes(rec.state_at(c))
    if rec.snapshot is not None:
        for key, value in vars(fresh).items():
            if key in rec.snapshot:
                verdict = same_attr(value, rec.snapshot[key])
                if verdict is False:
                    return False, "internal attribute '{}' differs after rebuilding".format(key)
    for t in range(c, min(len(mv), c + 3 * K)):
        if t % K == 0 and t > c and not np.array_equal(fresh.cube_position, rec.state_at(t)):
            return False, "positions differ from the original run at move {}".format(t)
        if fresh.apply_single_update_step(*mv[t]) == 0:
            return False, "a legal original move is illegal after rebuilding (move {})".format(t)
    return True, "ok"


def find_excursions(mv, hist):
    by_cube = {}
    for t, ((c, m), (b, a)) in enumerate(zip(mv, hist)):
        by_cube.setdefault(c, []).append((t, b, a))
    groups = []
    for c, lst in by_cube.items():
        seen = {lst[0][1]: -1}
        for k, (t, b, a) in enumerate(lst):
            if a in seen:
                i = seen[a]
                groups.append([lst[q][0] for q in range(i + 1, k + 1)])
                seen = {a: k}
            else:
                seen[a] = k
    return groups


tests = 0
simulated = 0


def test(mv, rec, removed):
    """Replay with `removed` (sorted indices) deleted. Returns (ok, converged-at index)."""
    global tests, simulated
    tests += 1
    first, last = removed[0], removed[-1]
    start = (first // K) * K
    cubes = ProgrammableCubes(rec.state_at(start))
    rem = set(removed)
    limit = last + args.max_tail * K
    for t in range(start, rec.L):
        if t > last and t % K == 0:
            if np.array_equal(cubes.cube_position, rec.state_at(t)):
                simulated += t - start
                return True, t
            if t > limit:
                simulated += t - start
                return False, None
        if t not in rem:
            c, m = mv[t]
            if cubes.apply_single_update_step(c, m) == 0:
                simulated += t - start
                return False, None
    simulated += rec.L - start
    if np.array_equal(cubes.cube_position, rec.final):
        return True, rec.L
    return False, None


def group_test(mv, rec, gs, deadline):
    accepted, converged = set(), 0
    stack = [gs]
    while stack and time.time() < deadline:
        cur = stack.pop()
        removed = accepted | {t for g in cur for t in g}
        ok, conv = test(mv, rec, sorted(removed))
        if ok:
            accepted, converged = removed, max(converged, conv)
        elif len(cur) > 1:
            mid = len(cur) // 2
            stack.append(cur[mid:])
            stack.append(cur[:mid])
    return accepted, converged


def without(mv, removed):
    return [x for t, x in enumerate(mv) if t not in removed]


t0 = time.time()
deadline = t0 + args.max_minutes * 60
rec = Record(moves, keep_object_at=(len(moves) // (2 * K)) * K if len(moves) >= 3 * K else None)
if not rec.ok:
    raise SystemExit("The input solution contains illegal moves; cannot prune safely.")
C0 = count_correct(rec.final)
L0 = len(moves)
print("input: L =", L0, "| C =", C0, "| stored f =", round(data.get("fitness", float("nan")), 5))
good, why = self_test(moves, rec)
print("self-test:", why)
if not good:
    raise SystemExit("Fast mode is not safe for this simulator. Use prune.py instead.")

rounds = 0
while time.time() < deadline:
    groups = find_excursions(moves, rec.hist)
    if not groups:
        break
    groups.sort(key=lambda g: g[0])
    accepted_all = set()
    barrier = 0
    i = 0
    while i < len(groups) and time.time() < deadline:
        chunk = []
        while i < len(groups) and len(chunk) < args.max_groups:
            g = groups[i]
            if g[0] < barrier:
                i += 1
                continue
            if chunk and g[-1] - chunk[0][0] > args.span:
                break
            chunk.append(g)
            i += 1
        if not chunk:
            continue
        acc, conv = group_test(moves, rec, chunk, deadline)
        if acc:
            accepted_all |= acc
            barrier = conv
    if not accepted_all:
        break
    moves = without(moves, accepted_all)
    rec = Record(moves)
    if not rec.ok:
        raise SystemExit("Internal error: pruned list is illegal. Use prune.py instead.")
    rounds += 1
    print("round {}: removed {} commands -> L = {} ({} groups, {} tests, {:.0f}M moves simulated, {:.0f} s)".format(
        rounds, len(accepted_all), len(moves), len(groups), tests, simulated / 1e6, time.time() - t0))

# Official evaluation
x = np.array([v for mv in moves for v in mv] + [-1])
f = udp.fitness(x)[0]
C_final = count_correct(udp.final_cube_positions)
print("\n=== PRUNED ({}) ===".format(args.problem))
print("L: {} -> {} | C: {} -> {} | f: {} -> {}".format(
    L0, len(moves), C0, C_final, round(data.get("fitness", float("nan")), 5), round(f, 5)))
print("time:", round(time.time() - t0, 1), "s | tests:", tests)

out = args.out or "solutions/{}_pruned.json".format(args.problem.lower())
os.makedirs("solutions", exist_ok=True)
better = True
if os.path.exists(out):
    with open(out, encoding="utf-8") as fh:
        better = f < json.load(fh)["fitness"]
if C_final < C0:
    print("WARNING: official C dropped; not saved. Run prune.py instead.")
elif better:
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"problem": args.problem, "fitness": f, "C": int(C_final), "L": len(moves),
                   "source": args.solution,
                   "chromosome": [int(v) for mv in moves for v in mv] + [-1]}, fh)
    print("saved", out)
else:
    print("not saved (not better than the stored file)")
