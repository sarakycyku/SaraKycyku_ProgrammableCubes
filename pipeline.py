"""pipeline.py: the whole fast pipeline in one command.
   many seeds of fast SA -> exact BFS repair -> rank by C -> prune the best K -> official check -> save.

Usage:  python pipeline.py JWST --seeds 100 --top 5 --workers 2 --prune-minutes 5
Saves solutions/<problem>_pipeline_best.json ONLY if better than the file that is already there,
and saves every officially scored pruned candidate under solutions/candidates/.
"""
import argparse, json, os, time, statistics
from multiprocessing import Pool

_ctx = None
P = {}


def init(problem, params):
    global _ctx, P
    from fast_sa import Context
    _ctx = Context(problem)
    P = params


def run(seed):
    from fast_sa import anneal
    from endgame import Endgame
    r = anneal(_ctx, seed, 50000000, P["t0"], P["t1"], P["p_mis"], P["p_zero"])
    moves = list(zip(r["cube"].tolist(), r["cmd"].tolist()))
    eg = Endgame(_ctx, moves, seed=seed)
    c = eg.repair()
    return seed, c, eg.moves


def prune_one(args):
    from fast_prune import Pruner
    seed, c, moves, minutes = args
    pruned = Pruner(_ctx).prune(moves, time.time() + minutes * 60, verbose=False)
    return seed, c, pruned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prune-minutes", type=float, default=30, help="cap per solution; pruning normally converges earlier")
    ap.add_argument("--t0", type=float, default=0.5)
    ap.add_argument("--t1", type=float, default=0.25)
    ap.add_argument("--p-mis", type=float, default=0.5)
    ap.add_argument("--p-zero", type=float, default=0.1)
    a = ap.parse_args()
    params = dict(t0=a.t0, t1=a.t1, p_mis=a.p_mis, p_zero=a.p_zero)
    t = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.problem, params)) as pool:
        res = pool.map(run, range(a.seed0, a.seed0 + a.seeds), chunksize=1)
        cs = [r[1] for r in res]
        print(f"{a.problem}: {a.seeds} seeds, {time.time()-t:.0f}s | C mean {statistics.mean(cs):.1f} max {max(cs)}", flush=True)
        res.sort(key=lambda r: (-r[1], len(r[2])))
        jobs = [(s, c, m, a.prune_minutes) for s, c, m in res[:a.top]]
        pruned = pool.map(prune_one, jobs, chunksize=1)
    from annealing_v5 import official_result
    ctx = _ctx_local = None
    from fast_sa import Context
    ctx = Context(a.problem)
    best = None
    os.makedirs("solutions/candidates", exist_ok=True)
    for seed, c, moves in pruned:
        f, co = official_result(ctx.udp, moves, ctx.udp.initial_cube_types, ctx.target_positions,
                                ctx.udp.target_cube_types, ctx.num_types)
        print(f"  seed {seed}: C {c} -> {co}, L {len(moves)}, f {f:.5f}")
        json.dump({"problem": a.problem, "fitness": f, "C": int(co), "L": len(moves), "seed": seed,
                   "chromosome": [int(v) for mv in moves for v in mv] + [-1]},
                  open(f"solutions/candidates/{a.problem.lower()}_seed{seed}.json", "w"))
        if best is None or f < best[0]:
            best = (f, co, moves, seed)
    f, co, moves, seed = best
    out = f"solutions/{a.problem.lower()}_pipeline_best.json"
    os.makedirs("solutions", exist_ok=True)
    old = json.load(open(out))["fitness"] if os.path.exists(out) else 1e9
    print(f"BEST {a.problem}: f={f:.5f} C={co} L={len(moves)} (seed {seed}) | total {time.time()-t:.0f}s")
    if f < old:
        json.dump({"problem": a.problem, "fitness": f, "C": int(co), "L": len(moves), "seed": seed,
                   "chromosome": [int(v) for mv in moves for v in mv] + [-1]}, open(out, "w"))
        print("saved", out)
    else:
        print("not saved: stored file is better", old)


if __name__ == "__main__":
    main()
