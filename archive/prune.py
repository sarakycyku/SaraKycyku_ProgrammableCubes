# prune.py: heq komandat e tepërta nga një zgjidhje e ruajtur, pa ulur C.
#
# Ideja: nese nje kub largohet nga nje pozite dhe me vone kthehet saktesisht aty, te gjitha
# levizjet e atij kubi ne ate interval (nje "ekskursion") mund te jene te kota. Nuk eshte e
# garantuar (kubet e tjere mund te kene mbeshtetur ne to), prandaj cdo heqje verifikohet me
# rilojtje te plote me simulatorin zyrtar. Per te shmangur shume rilojtje, grupet provohen
# se bashku dhe ndahen ne gjysma kur dështojnë (group testing).
#
# Perdorimi (nga dosja ku eshte programmable_cubes_UDP.py):
#   python prune.py JWST solutions/jwst_annealing_v3.json --max-minutes 20
#   python prune.py ISS solutions/iss_annealing_v2.json --max-minutes 5
import argparse
import json
import os
import time
import numpy as np
from programmable_cubes_UDP import programmable_cubes_UDP, ProgrammableCubes

ap = argparse.ArgumentParser()
ap.add_argument("problem")
ap.add_argument("solution", help="JSON me kromozomin")
ap.add_argument("--out", default=None, help="skedari dalje (default solutions/<problem>_pruned.json)")
ap.add_argument("--max-minutes", type=float, default=20.0)
args = ap.parse_args()

udp = programmable_cubes_UDP(args.problem)
s = udp.setup
init_pos = np.load("{}/Initial_Config.npy".format(s["path"]))
types = udp.initial_cube_types
tgt_pos = udp.target_cube_positions
tgt_types = udp.target_cube_types
n = len(init_pos)
target_type_at = {tuple(p): int(t) for p, t in zip(tgt_pos.tolist(), tgt_types.tolist())}

with open(args.solution) as fh:
    data = json.load(fh)
chrom = data["chromosome"]
moves = []
for i in range(0, len(chrom) - 1, 2):
    if chrom[i] == -1:
        break
    moves.append((int(chrom[i]), int(chrom[i + 1])))


def count_correct(positions):
    return sum(1 for i in range(n)
               if target_type_at.get(tuple(positions[i].tolist())) == int(types[i]))


def replay(mv, record=False):
    """Rilojtje nga e para. Kthen (ok, pozicionet finale, historia). ok=False nese ndonje leviz. eshte e paligjshme."""
    cubes = ProgrammableCubes(init_pos)
    hist = [] if record else None
    for c, m in mv:
        before = cubes.cube_position[c].copy() if record else None
        if cubes.apply_single_update_step(c, m) == 0:
            return False, None, None
        if record:
            hist.append((before, cubes.cube_position[c].copy()))
    return True, cubes.cube_position, hist


def find_excursions(mv, hist):
    """Grupe indeksesh levizjesh: per secilin kub, intervale ku kthehet ne nje pozite te mepareshme."""
    by_cube = {}
    for t, ((c, m), (b, a)) in enumerate(zip(mv, hist)):
        by_cube.setdefault(c, []).append((t, tuple(b.tolist()), tuple(a.tolist())))
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


def without(mv, removed):
    return [x for t, x in enumerate(mv) if t not in removed]


replays = 0


def valid(mv, C_min):
    global replays
    replays += 1
    ok, final, _ = replay(mv)
    return ok and count_correct(final) >= C_min


def try_groups(mv, groups, C_min, deadline):
    """Group testing: provo te heqesh te gjitha grupet; nese dështon, nda ne gjysma."""
    accepted = set()
    stack = [groups]
    while stack and time.time() < deadline:
        gs = stack.pop()
        removed = accepted | {t for g in gs for t in g}
        if valid(without(mv, removed), C_min):
            accepted = removed
        elif len(gs) > 1:
            mid = len(gs) // 2
            stack.append(gs[mid:])
            stack.append(gs[:mid])
    return accepted


t0 = time.time()
deadline = t0 + args.max_minutes * 60
ok, final, hist = replay(moves, record=True)
if not ok:
    raise SystemExit("The input solution contains illegal moves; cannot prune safely.")
C0 = count_correct(final)
L0 = len(moves)
print("input: L =", L0, "| C =", C0, "| stored f =", round(data.get("fitness", float("nan")), 5))

rounds = 0
while time.time() < deadline:
    groups = find_excursions(moves, hist)
    if not groups:
        break
    groups.sort(key=len)
    removed = try_groups(moves, groups, C0, deadline)
    if not removed:
        break
    moves = without(moves, removed)
    ok, final, hist = replay(moves, record=True)
    rounds += 1
    print("round {}: removed {} commands -> L = {} ({} candidate groups, {} replays, {:.0f} s)".format(
        rounds, len(removed), len(moves), len(groups), replays, time.time() - t0))

# Vleresimi zyrtar
x = np.array([v for mv in moves for v in mv] + [-1])
f = udp.fitness(x)[0]
final = udp.final_cube_positions
C_final = count_correct(final)
print("\n=== PRUNED ({}) ===".format(args.problem))
print("L: {} -> {} | C: {} -> {} | f: {} -> {}".format(
    L0, len(moves), C0, C_final, round(data.get("fitness", float("nan")), 5), round(f, 5)))
print("time:", round(time.time() - t0, 1), "s | replays:", replays)

out = args.out or "solutions/{}_pruned.json".format(args.problem.lower())
os.makedirs("solutions", exist_ok=True)
better = True
if os.path.exists(out):
    with open(out) as fh:
        better = f < json.load(fh)["fitness"]
if better and C_final >= C0:
    with open(out, "w") as fh:
        json.dump({"problem": args.problem, "fitness": f, "C": int(C_final), "L": len(moves),
                   "source": args.solution,
                   "chromosome": [int(v) for mv in moves for v in mv] + [-1]}, fh)
    print("saved", out)
else:
    print("not saved (not better than the stored file, or C dropped)")