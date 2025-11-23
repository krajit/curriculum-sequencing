#!/usr/bin/env python3
"""course_selector.py
Clean implementation for curriculum sequencing.
Usage:
    python3 course_selector.py --csv connectivityMatrix.csv --nc 2 --alpha 0.5 --enumeration_limit 50000
"""
import argparse, itertools as it, numpy as np, pandas as pd, random
from collections import deque

def load_connectivity_from_csv(path):
    df = pd.read_csv(path, index_col=0)
    courses = list(df.columns)
    C = df.values.astype(int)
    return C, courses

def zero_indegree_courses(C, remaining):
    subC = C[np.ix_(remaining, remaining)]
    col_sums = np.sum(subC, axis=0)
    zeros = [remaining[i] for i, val in enumerate(col_sums) if val == 0]
    return zeros

def enumerate_feasible(C, nc, limit=None):
    n = C.shape[0]
    all_indices = list(range(n))
    results = []
    truncated = False
    def backtrack(prefix):
        nonlocal results, truncated
        if limit is not None and len(results) >= limit:
            truncated = True
            return
        if len(prefix) == n:
            results.append(prefix.copy()); return
        remaining = [i for i in all_indices if i not in prefix]
        zeros = zero_indegree_courses(C, remaining)
        if len(zeros) < nc: return
        for combo in it.combinations(zeros, nc):
            prefix.extend(combo); backtrack(prefix)
            for _ in combo: prefix.pop()
            if truncated: return
    backtrack([])
    return results, truncated

def compute_TIR(C, ordering, nc, alpha=0.5):
    sem_map = {}
    for pos, course in enumerate(ordering):
        sem = (pos // nc) + 1; sem_map[course] = sem
    tir = 0.0; n_nodes = C.shape[0]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if C[i, j] == 1:
                sA = sem_map[i]; sB = sem_map[j]
                if sB >= sA + 1:
                    gap = sB - sA - 1; tir += (alpha ** gap)
    return tir

def topological_order_to_curriculum(C, nc):
    n = C.shape[0]; in_deg = np.sum(C, axis=0).tolist()
    queue = deque([i for i in range(n) if in_deg[i] == 0]); topo = []
    while queue:
        v = queue.popleft(); topo.append(v)
        for u in range(n):
            if C[v, u] == 1:
                in_deg[u] -= 1
                if in_deg[u] == 0: queue.append(u)
    if len(topo) != n: return None
    return topo

def sample_random_feasible(C, nc, sample_size=1000, seed=42):
    random.seed(seed); n = C.shape[0]; samples = []
    for _ in range(sample_size):
        ordering = []; remaining = set(range(n)); feasible = True
        while remaining:
            zeros = zero_indegree_courses(C, list(remaining))
            if len(zeros) < nc: feasible = False; break
            combo = random.sample(zeros, nc); ordering.extend(combo)
            for c in combo: remaining.remove(c)
        if feasible and len(ordering) == n: samples.append(ordering)
    return samples

def ordering_to_semesters(ordering, courses, nc):
    n = len(ordering); sems = []
    for s in range(0, n, nc):
        chunk = ordering[s:s+nc]; sems.append([courses[i] for i in chunk])
    return sems

def save_top_curriculum(ordering, courses, nc, out_csv="top_curriculum_by_TIR.csv"):
    sems = ordering_to_semesters(ordering, courses, nc); rows = []
    for i, sem in enumerate(sems, start=1):
        for c in sem: rows.append({"Semester": f"Sem{i}", "Course_Name": c})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Curriculum sequencing tool")
    parser.add_argument("--csv", type=str, default="connectivityMatrix.csv")
    parser.add_argument("--nc", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--enumeration_limit", type=int, default=50000)
    parser.add_argument("--sample_random", action="store_true")
    parser.add_argument("--random_samples", type=int, default=1000)
    args = parser.parse_args()

    C, courses = load_connectivity_from_csv(args.csv); n = C.shape[0]
    print(f"Loaded {n} courses from {args.csv}. nc = {args.nc}")
    feasible_list, truncated = enumerate_feasible(C, args.nc, limit=args.enumeration_limit)
    if truncated: print(f"Enumeration truncated after {args.enumeration_limit} curricula.")
    print(f"Enumerated {len(feasible_list)} feasible curricula. Truncated = {truncated}")

    if len(feasible_list) == 0 and args.sample_random:
        print("Attempting random feasible sampling...")
        feasible_list = sample_random_feasible(C, args.nc, sample_size=args.random_samples)
        print(f"Sampled {len(feasible_list)} feasible curricula.")

    tir_scores = []
    for ordering in feasible_list:
        tir = compute_TIR(C, ordering, args.nc, alpha=args.alpha); tir_scores.append((tir, ordering))
    tir_scores.sort(reverse=True, key=lambda x: x[0])

    results = {}
    if tir_scores:
        best_tir, best_ordering = tir_scores[0]; worst_tir, worst_ordering = tir_scores[-1]
        mean_tir = float(np.mean([t for t,_ in tir_scores])); std_tir = float(np.std([t for t,_ in tir_scores]))
        results.update({"best_tir": best_tir, "worst_tir": worst_tir, "mean_tir": mean_tir, "std_tir": std_tir, "count": len(tir_scores)})
        print("Best TIR =", best_tir)
        save_top_curriculum(best_ordering, courses, args.nc)
    else:
        print("No TIR scores computed.")

    topo = topological_order_to_curriculum(C, args.nc)
    if topo is not None:
        topo_tir = compute_TIR(C, topo, args.nc, alpha=args.alpha); results["topo_tir"] = topo_tir
        print("Topological baseline TIR =", topo_tir)

    random_samples = sample_random_feasible(C, args.nc, sample_size=args.random_samples)
    if random_samples:
        rand_tirs = [compute_TIR(C, ordn, args.nc, alpha=args.alpha) for ordn in random_samples]
        results["random_mean"] = float(np.mean(rand_tirs)); results["random_std"] = float(np.std(rand_tirs))
        print("Random feasible baseline mean TIR =", results["random_mean"])

    import json
    with open("tir_summary.json", "w") as fh: json.dump(results, fh, indent=2)
    print("Saved tir_summary.json and (if available) top_curriculum_by_TIR.csv")

if __name__ == "__main__": main()
