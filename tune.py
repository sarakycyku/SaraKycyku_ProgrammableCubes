"""Parameter ablation: run several configurations on several seeds and print a mean/std table.

Use it on the CHEAP problems first (ISS: about 1 minute for everything with 4 workers), then on JWST,
and only then run the long Enterprise runs with the winning settings.

Examples (from the project folder):
  python tune.py ISS  --algo sa    --trials 1000000 --seeds 0-7 --workers 4
  python tune.py JWST --algo sa    --trials 600000  --seeds 0-3 --workers 4
  python tune.py ISS  --algo tabu  --trials 200000  --seeds 0-7 --workers 4
  python tune.py ISS  --algo sa --only base,dir03     # only some configurations

--trials is the number of trials for sa (annealing_v5.py) and iterations for tabu (tabu_search_v2.py).
The first configuration ("base") is the reference; the table says whether a configuration differs
from it by more than about 2 standard errors. Results: results/tune_<problem>_<algo>_<label>.csv
"""
import argparse
import csv
import os
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CONFIGS = {
    "sa": {
        "base": "",                       # v3 baseline: p-mis 0.5, p-zero 0.3, no directed proposals
        "pmis07": "--p-mis 0.7",
        "dir03": "--p-dir 0.3",
        "dir06": "--p-dir 0.6",
        "zero01": "--p-zero 0.1",
        "zero05": "--p-zero 0.05",
        "T35_20": "--t0 0.35 --t1 0.2",
        "T60_30": "--t0 0.6 --t1 0.3",
    },
    "tabu": {
        "base": "",                       # candidates 16, tenure 30, patience 5000, p-dir 0.5
        "ten10": "--tenure 10",
        "ten100": "--tenure 100",
        "cand8": "--candidates 8",
        "cand32": "--candidates 32",
        "pat1500": "--patience 1500",
        "pat20000": "--patience 20000",
        "dir0": "--p-dir 0",
    },
}
SCRIPT = {"sa": "annealing_v5.py", "tabu": "tabu_search_v2.py"}
TRIALS_OPTION = {"sa": "--trials", "tabu": "--iters"}
CSV_FILE = {"sa": "results/annealing_v5_runs.csv", "tabu": "results/tabu_v2_runs.csv"}


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
    parser.add_argument("--algo", choices=["sa", "tabu"], default="sa")
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--seeds", default="0-7")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default="", help="comma-separated configuration names")
    parser.add_argument("--label", default=time.strftime("%m%d%H%M"),
                        help="label for this experiment (part of the tags)")
    parser.add_argument("--extra", default="", help="options added to every run, in quotes")
    args = parser.parse_args()

    configs = CONFIGS[args.algo]
    names = [name for name in configs if not args.only or name in args.only.split(",")]
    if args.only and "base" not in names:
        names.insert(0, "base")
    seeds = parse_seeds(args.seeds)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT[args.algo])
    jobs = []
    for name in names:
        for seed in seeds:
            command = ([sys.executable, script, args.problem, TRIALS_OPTION[args.algo],
                        str(args.trials), "--seed", str(seed), "--tag", f"tune{args.label}_{name}"]
                       + shlex.split(configs[name]) + shlex.split(args.extra))
            jobs.append((name, seed, command))
    print(f"{len(names)} configurations x {len(seeds)} seeds = {len(jobs)} runs, "
          f"{args.workers} workers, label {args.label}", flush=True)

    def run(job):
        name, seed, command = job
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return name, seed, result.returncode, result.stderr[-400:]

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, seed, code, error in pool.map(run, jobs):
            done += 1
            if code != 0:
                print(f"  FAILED {name} seed {seed} (exit {code}): {error}", flush=True)
            elif done % max(1, len(jobs) // 10) == 0:
                print(f"  {done}/{len(jobs)} runs done ({time.time() - started:.0f} s)", flush=True)

    rows = {}
    with open(CSV_FILE[args.algo], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["problem"] == args.problem and row["tag"].startswith(f"tune{args.label}_"):
                rows.setdefault(row["tag"].split("_", 1)[1], []).append(row)

    summary = []
    for name in names:
        runs = rows.get(name, [])
        if not runs:
            continue
        f = [float(r["fitness"]) for r in runs]
        summary.append({
            "config": name, "options": configs[name] or "(defaults)", "runs": len(f),
            "mean_f": statistics.mean(f),
            "std_f": statistics.stdev(f) if len(f) > 1 else 0.0,
            "best_f": min(f),
            "mean_C": statistics.mean(int(r["C"]) for r in runs),
            "mean_L": statistics.mean(int(r["L"]) for r in runs),
            "mean_s": statistics.mean(float(r["seconds"]) for r in runs),
        })
    if not summary:
        raise SystemExit("no results found; did all runs fail?")
    base = summary[0] if summary[0]["config"] == "base" else None
    for item in summary:
        item["verdict"] = ""
        if base is not None and item is not base:
            se = (item["std_f"] ** 2 / item["runs"] + base["std_f"] ** 2 / base["runs"]) ** 0.5
            diff = item["mean_f"] - base["mean_f"]  # negative = better than base
            if se > 0 and abs(diff) > 2 * se:
                item["verdict"] = "BETTER" if diff < 0 else "WORSE"
            else:
                item["verdict"] = "same"

    summary.sort(key=lambda item: item["mean_f"])
    print(f"\n{args.problem} {args.algo}: {args.trials} {TRIALS_OPTION[args.algo][2:]}, seeds {args.seeds} "
          f"(lower f is better; verdict is relative to base)")
    print(f"{'config':10s} {'mean f':>8s} {'std':>7s} {'best f':>8s} {'mean C':>7s} {'mean L':>8s} "
          f"{'sec':>6s}  verdict  options")
    for item in summary:
        print(f"{item['config']:10s} {item['mean_f']:8.4f} {item['std_f']:7.4f} {item['best_f']:8.4f} "
              f"{item['mean_C']:7.1f} {item['mean_L']:8.0f} {item['mean_s']:6.1f}  "
              f"{item['verdict']:7s}  {item['options']}")
    os.makedirs("results", exist_ok=True)
    out = f"results/tune_{args.problem.lower()}_{args.algo}_{args.label}.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("saved", out)


if __name__ == "__main__":
    main()
