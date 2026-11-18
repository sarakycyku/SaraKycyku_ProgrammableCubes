"""Run annealing_v5.py (or v4 with --script) on several seeds in parallel, then restart from the best result.

Example (from the project folder):
  python run_parallel.py ISS --seeds 0-7 --workers 4 --trials 1000000 --rounds 2
  python run_parallel.py Enterprise --seeds 0-3 --workers 4 --trials 2000000 --rounds 1 \
      --extra "--p-mis 0.7 --p-zero 0.05"

Round 0 runs all seeds from scratch (or from --start). Each later round restarts every seed
from the best solution of the previous round with a lower temperature range.
Solutions: solutions/<problem>_annealing_<tag>r<round>.json (best of each round).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


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
    parser.add_argument("--seeds", default="0-3", help="for example 0-7 or 0,5,9")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=600000, help="trials in round 0")
    parser.add_argument("--t0", type=float, default=0.5)
    parser.add_argument("--t1", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=1,
                        help="1 = only round 0; 2 = one restart round, and so on")
    parser.add_argument("--restart-trials", type=int, default=400000)
    parser.add_argument("--restart-t0", type=float, default=0.35)
    parser.add_argument("--restart-t1", type=float, default=0.2)
    parser.add_argument("--start", default=None, help="solution JSON to begin round 0 from")
    parser.add_argument("--script", default="annealing_v5.py",
                        help="annealing script to run (annealing_v4.py or annealing_v5.py)")
    parser.add_argument("--tag", default="chain")
    parser.add_argument("--extra", default="",
                        help="extra options passed to the annealing script, in quotes")
    args = parser.parse_args()

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.script)
    seeds = parse_seeds(args.seeds)
    extra = shlex.split(args.extra)
    previous_best = args.start

    def run_one(job):
        command, seed = job
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return seed, result.returncode, result.stderr[-500:]

    for round_index in range(args.rounds):
        tag = f"{args.tag}r{round_index}"
        trials = args.trials if round_index == 0 else args.restart_trials
        t0 = args.t0 if round_index == 0 else args.restart_t0
        t1 = args.t1 if round_index == 0 else args.restart_t1
        jobs = []
        for seed in seeds:
            command = [sys.executable, script, args.problem, "--trials", str(trials),
                       "--seed", str(seed), "--t0", str(t0), "--t1", str(t1), "--tag", tag]
            if previous_best:
                command += ["--start", previous_best]
            jobs.append((command + extra, seed))
        print(f"round {round_index}: {len(seeds)} seeds, {trials} trials, T {t0} -> {t1}"
              + (f", start {previous_best}" if previous_best else ""), flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for seed, code, error in pool.map(run_one, jobs):
                if code != 0:
                    print(f"  seed {seed} FAILED (exit {code}): {error}", flush=True)
        best_path = f"solutions/{args.problem.lower()}_annealing_{tag}.json"
        if not os.path.exists(best_path):
            raise SystemExit(f"no solution file {best_path}; all runs failed?")
        with open(best_path, encoding="utf-8") as source:
            best = json.load(source)
        print(f"  best after round {round_index}: f={best['fitness']:.5f} "
              f"C={best['C']} L={best['L']} -> {best_path}", flush=True)
        previous_best = best_path

    print("done. Next step: python prune_fast.py", args.problem, previous_best)


if __name__ == "__main__":
    main()
