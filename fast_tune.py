"""fast_tune.py: parameter grid search for fast_sa.py with many seeds, in-process (no start-up cost).

Example:
  python fast_tune.py ISS --seeds 0-9 --grid "t0=0.5,0.3;t1=0.25,0.1;p_zero=0.1,0.03" --workers 2
Grid keys: t0, t1, p_zero, p_mis, cap_frac, trials, rematch_every. Missing keys use the defaults.
Prints a table sorted by mean f (lower is better) and saves results/fast_tune_<problem>_<label>.csv.
"""
import argparse
import csv
import itertools
import os
import statistics
import time
from multiprocessing import Pool

DEFAULTS = {"t0": 0.5, "t1": 0.25, "p_zero": 0.3, "p_mis": 0.5, "cap_frac": 1.0,
            "trials": 50000000, "rematch_every": 200}
_ctx = None


def _init(problem):
    global _ctx
    from fast_sa import Context
    _ctx = Context(problem)


def _run(job):
    from fast_sa import anneal
    config_id, params, seed = job
    result = anneal(_ctx, seed, int(params["trials"]), params["t0"], params["t1"], params["p_mis"],
                    params["p_zero"], params["cap_frac"], int(params["rematch_every"]))
    return config_id, seed, result["f"], result["C"], result["L"], result["seconds"], result["trials_done"]


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("problem")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--grid", default="", help='e.g. "t0=0.5,0.3;p_zero=0.3,0.1"')
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label", default=time.strftime("%m%d%H%M"))
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    axes = {}
    for part in filter(None, args.grid.split(";")):
        key, values = part.split("=")
        if key not in DEFAULTS:
            raise SystemExit(f"unknown key {key}; use one of {list(DEFAULTS)}")
        axes[key] = [float(v) for v in values.split(",")]
    keys = list(axes)
    configs = []
    for combo in itertools.product(*[axes[k] for k in keys]) if keys else [()]:
        params = dict(DEFAULTS)
        params.update(dict(zip(keys, combo)))
        configs.append(params)
    seeds = parse_seeds(args.seeds)
    jobs = [(i, cfg, seed) for i, cfg in enumerate(configs) for seed in seeds]
    print(f"{len(configs)} configurations x {len(seeds)} seeds = {len(jobs)} runs", flush=True)
    started = time.time()
    rows = {}
    with Pool(args.workers, initializer=_init, initargs=(args.problem,)) as pool:
        for count, (cid, seed, f, c, l, sec, k) in enumerate(pool.imap_unordered(_run, jobs, chunksize=1), 1):
            rows.setdefault(cid, []).append((f, c, l, sec, k))
            if count % max(1, len(jobs) // 10) == 0:
                print(f"  {count}/{len(jobs)} ({time.time() - started:.0f} s)", flush=True)

    summary = []
    for cid, runs in rows.items():
        fs = [r[0] for r in runs]
        summary.append({
            **{k: configs[cid][k] for k in keys},
            "runs": len(fs), "mean_f": statistics.mean(fs),
            "std_f": statistics.stdev(fs) if len(fs) > 1 else 0.0, "best_f": min(fs),
            "mean_C": statistics.mean(r[1] for r in runs), "mean_L": statistics.mean(r[2] for r in runs),
            "mean_s": statistics.mean(r[3] for r in runs), "mean_trials": statistics.mean(r[4] for r in runs),
        })
    summary.sort(key=lambda r: r["mean_f"])
    header = keys + ["runs", "mean_f", "std_f", "best_f", "mean_C", "mean_L", "mean_s", "mean_trials"]
    print("\n" + "  ".join(f"{h:>9s}" for h in header))
    for row in summary[:args.top]:
        print("  ".join(f"{row[h]:9.4g}" if isinstance(row[h], float) else f"{row[h]:9}" for h in header))
    os.makedirs("results", exist_ok=True)
    out = f"results/fast_tune_{args.problem.lower()}_{args.label}.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(summary)
    print("saved", out, f"| total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
