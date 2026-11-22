"""Many fast-SA seeds, each followed by exact BFS repair; prints the distribution and keeps the best."""
import sys, time, json, statistics
import numpy as np
from multiprocessing import Pool
problem = sys.argv[1]; seeds = int(sys.argv[2]); extra = dict(a.split("=") for a in sys.argv[3:])
cfg = dict(t0=0.5, t1=0.25, p_zero=0.1, p_mis=0.5, kicks=0)
cfg.update({k: float(v) for k, v in extra.items()})
_ctx = None
def init():
    global _ctx
    from fast_sa import Context
    _ctx = Context(problem)
def run(seed):
    from fast_sa import anneal
    from endgame import Endgame
    r = anneal(_ctx, seed, 50000000, cfg["t0"], cfg["t1"], cfg["p_mis"], cfg["p_zero"])
    moves = list(zip(r["cube"].tolist(), r["cmd"].tolist()))
    eg = Endgame(_ctx, moves, seed=seed)
    c = eg.repair()
    if cfg["kicks"]:
        c, _ = eg.kick_and_repair(int(cfg["kicks"]), 2, 3, verbose=False)
    return seed, r["C"], c, len(moves), len(eg.moves), _ctx.f_of(c, len(eg.moves)), eg.moves
if __name__ == "__main__":
    t = time.time()
    with Pool(2, initializer=init) as pool:
        res = pool.map(run, range(seeds), chunksize=1)
    cs0 = [r[1] for r in res]; cs = [r[2] for r in res]; fs = [r[5] for r in res]
    print(f"{problem}: {seeds} seeds in {time.time()-t:.0f}s")
    print(f"  after SA     : C mean {statistics.mean(cs0):.1f} max {max(cs0)}")
    print(f"  after repair : C mean {statistics.mean(cs):.1f} max {max(cs)} | f mean {statistics.mean(fs):.4f} best {min(fs):.5f}")
    best = min(res, key=lambda r: r[5])
    print(f"  best seed {best[0]}: C {best[1]} -> {best[2]}, L {best[3]} -> {best[4]}, f {best[5]:.5f}")
    json.dump({"problem": problem, "fitness": best[5], "C": best[2], "L": best[4],
               "chromosome": [int(v) for mv in best[6] for v in mv] + [-1]}, open(f"solutions/{problem.lower()}_multi_best.json", "w"))
