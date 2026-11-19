"""Compare saved solutions of different algorithms and plot the results.

Edit the RUNS dictionary below: for each problem, list "label": "path to solution JSON".
Missing files are skipped. Every solution is replayed with the official simulator, so the
numbers (f, C, L) are recomputed and do not depend on what is stored in the JSON.

Run from the project folder:
  python compare.py                # all problems found in RUNS
  python compare.py ISS JWST       # only these
Outputs a new versioned report so existing results are never overwritten:
results/comparison_v5.csv, results/compare_v5_<problem>.png, results/compare_v5_all.png
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from programmable_cubes_UDP import programmable_cubes_UDP

# Order = order in the plots. Put weaker/simpler algorithms first.
RUNS = {
    "ISS": {
        "Greedy v1": "solutions/iss_greedy_v1.json",
        "Hill climbing": "solutions/iss_annealing_hill.json",
        "Tabu search": "solutions/iss_tabu_v1.json",
        "Annealing v2": "solutions/iss_annealing_v2.json",
        "Annealing + restart (v3)": "solutions/iss_annealing_v3.json",
        "Annealing v4": "solutions/iss_annealing_v4.json",
        "SA v5": "solutions/iss_annealing_f5r0.json",
        "Tabu v2": "solutions/iss_tabu_ft2.json",
        "SA v5 + pruning": "solutions/iss_final_sa_pruned.json",
        "Tabu v2 + pruning": "solutions/iss_final_tabu_pruned.json",
        "Annealing + pruning": "solutions/iss_pruned.json",
    },
    "JWST": {
        "Hill climbing": "solutions/jwst_annealing_hill.json",
        "Tabu search": "solutions/jwst_tabu_v1.json",
        "Annealing v2": "solutions/jwst_annealing_v2.json",
        "Annealing + restart (v3)": "solutions/jwst_annealing_v3.json",
        "Annealing v4": "solutions/jwst_annealing_v4.json",
        "SA v5": "solutions/jwst_annealing_f5r0.json",
        "Tabu v2": "solutions/jwst_tabu_ft2.json",
        "SA v5 + pruning": "solutions/jwst_final_sa_pruned.json",
        "Tabu v2 + pruning": "solutions/jwst_final_tabu_pruned.json",
        "Annealing + pruning": "solutions/jwst_pruned.json",
    },
    "Enterprise": {
        "Hill climbing": "solutions/enterprise_annealing_hill.json",
        "Tabu search": "solutions/enterprise_tabu_v1.json",
        "Annealing (main)": "solutions/enterprise_annealing_main_seed_0.json",
        "Annealing v4": "solutions/enterprise_annealing_v4.json",
        "SA v5": "solutions/enterprise_annealing_f5r0.json",
        "Tabu v2": "solutions/enterprise_tabu_ft2.json",
        "SA v5 + pruning": "solutions/enterprise_final_sa_pruned.json",
        "Tabu v2 + pruning": "solutions/enterprise_final_tabu_pruned.json",
        "Annealing + pruning": "solutions/enterprise_pruned.json",
    },
}


def evaluate(problem, path, cache):
    if problem not in cache:
        cache[problem] = programmable_cubes_UDP(problem)
    udp = cache[problem]
    with open(path, encoding="utf-8") as source:
        chromosome = np.array(json.load(source)["chromosome"], dtype=int)
    fitness = float(udp.fitness(chromosome)[0])
    types = udp.initial_cube_types
    final = udp.final_cube_positions
    correct = 0
    for cube_type in range(udp.setup["num_cube_types"]):
        final_set = {tuple(p) for p in final[types == cube_type].tolist()}
        correct += sum(tuple(p) in final_set for p in
                       udp.target_cube_positions[udp.target_cube_types == cube_type].tolist())
    length = 0
    for i in range(0, len(chromosome) - 1, 2):
        if chromosome[i] == -1:
            break
        length += 1
    return fitness, correct, length, len(types), udp.setup["max_cmds"]


def plot_problem(axes, problem, rows):
    labels = [r["label"] for r in rows]
    scores = [-r["f"] for r in rows]  # higher is better
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))
    ax_bar, ax_tr = axes
    bars = ax_bar.bar(range(len(rows)), scores, color=colors)
    best = max(range(len(rows)), key=lambda i: scores[i])
    bars[best].set_edgecolor("black")
    bars[best].set_linewidth(2)
    for i, value in enumerate(scores):
        ax_bar.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_xticks(range(len(rows)))
    ax_bar.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylabel("score = -f (higher is better)")
    ax_bar.set_title(f"{problem}: score")
    ax_bar.set_ylim(0, max(1.0, max(scores) * 1.1))

    for row, color in zip(rows, colors):
        ax_tr.scatter(row["L"], 100 * row["C"] / row["n"], s=70, color=color, label=row["label"])
    ax_tr.set_xlabel("commands used (L)")
    ax_tr.set_ylabel("correct cubes (%)")
    ax_tr.set_title(f"{problem}: correctness vs. commands")
    ax_tr.set_ylim(0, 102)
    ax_tr.grid(alpha=0.3)
    ax_tr.legend(fontsize=7, loc="lower right")


def main():
    wanted = sys.argv[1:] or list(RUNS)
    cache, results, plot_results = {}, {}, {}
    for problem in wanted:
        rows = []
        for label, path in RUNS.get(problem, {}).items():
            if not os.path.exists(path):
                print(f"skip {problem} / {label}: {path} not found")
                continue
            f, c, l, n, max_cmds = evaluate(problem, path, cache)
            rows.append({"problem": problem, "label": label, "f": f, "C": c, "L": l,
                         "n": n, "max_cmds": max_cmds, "file": path})
            print(f"{problem:11s} {label:28s} f={f:8.5f}  C={c}/{n}  L={l}")
        unique, seen = [], {}
        for row in rows:
            key = (round(row["f"], 5), row["C"], row["L"])
            if key in seen:
                print(f"note: {problem} / {row['label']} is identical to '{seen[key]}' "
                      f"(same f, C, L); dropped from the plots. Check that the two files differ.")
                continue
            seen[key] = row["label"]
            unique.append(row)
        if unique:
            results[problem] = rows
            plot_results[problem] = unique
    if not results:
        raise SystemExit("nothing to compare: check the paths in RUNS")

    os.makedirs("results", exist_ok=True)
    with open("results/comparison_v5.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["problem", "algorithm", "f", "C", "n", "L", "file"])
        for rows in results.values():
            for r in rows:
                writer.writerow([r["problem"], r["label"], round(r["f"], 5), r["C"], r["n"], r["L"], r["file"]])

    for problem, rows in plot_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plot_problem(axes, problem, rows)
        fig.tight_layout()
        fig.savefig(f"results/compare_v5_{problem.lower()}.png", dpi=150)
        plt.close(fig)

    fig, all_axes = plt.subplots(len(plot_results), 2, figsize=(12, 4.5 * len(plot_results)), squeeze=False)
    for axes, (problem, rows) in zip(all_axes, plot_results.items()):
        plot_problem(axes, problem, rows)
    fig.tight_layout()
    fig.savefig("results/compare_v5_all.png", dpi=150)
    print("saved results/comparison_v5.csv and results/compare_v5_*.png")


if __name__ == "__main__":
    main()
