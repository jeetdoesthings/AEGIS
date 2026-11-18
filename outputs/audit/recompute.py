"""Audit script: recompute all headline numbers from raw results_*.json.

Recomputes ACC/BWT/NEW (mean +- std with ddof=0 and ddof=1) for every method
on all three benchmarks, plus per-seed paired differences vs fine-tuning.
Writes outputs/audit/recomputed_numbers.csv
"""
import json
import numpy as np
import csv
import os

FILES = {
    "perm5": [
        "outputs/phase1/final_perm5.json",
        "outputs/phase3/final_repaired_perm5.json",
    ],
    "perm10": [
        "outputs/phase2/final_perm10.json",
        "outputs/phase3/final_repaired_perm10.json",
    ],
    "split": [
        "outputs/phase2/final_split.json",
        "outputs/phase3/final_repaired_splitmnist.json",
    ],
}

rows = []
for bench, files in FILES.items():
    data = {}
    for f in files:
        if not os.path.exists(f):
            print(f"MISSING: {f}")
            continue
        with open(f) as fh:
            d = json.load(fh)
        for m, rs in d.items():
            if m in data:
                print(f"DUPLICATE method {m} across files for {bench}")
            data[m] = rs
    if "finetune" not in data:
        print(f"{bench}: no finetune reference!")
        continue
    ft_accs = None
    for m, rs in data.items():
        if any(r.get("diverged", False) for r in rs):
            rows.append([bench, m, "DIVERGED"] + [""] * 9)
            continue
        accs = np.array([r["ACC"] for r in rs]) * 100
        bwts = np.array([r["BWT"] for r in rs]) * 100
        news = np.array([r["NEW"] for r in rs]) * 100
        n = len(accs)
        if m == "finetune":
            ft_accs = accs
            ft_bwts = bwts
        row = [bench, m, n,
               f"{accs.mean():.2f}", f"{accs.std(ddof=0):.2f}", f"{accs.std(ddof=1):.2f}",
               f"{bwts.mean():.2f}", f"{bwts.std(ddof=0):.2f}", f"{bwts.std(ddof=1):.2f}",
               f"{news.mean():.2f}"]
        rows.append(row)
        if ft_accs is not None and m != "finetune":
            dacc = accs - ft_accs
            dbwt = bwts - ft_bwts
            rows.append([bench, m + " (paired vs FT)", n,
                         f"{dacc.mean():+.2f}", f"{dacc.std(ddof=0):.2f}", f"{dacc.std(ddof=1):.2f}",
                         f"{dbwt.mean():+.2f}", f"{dbwt.std(ddof=0):.2f}", f"{dbwt.std(ddof=1):.2f}",
                         ""])
    # tie check: intervals overlap?
    ft = data["finetune"]
    ft_acc = np.array([r["ACC"] for r in ft]) * 100
    ft_lo, ft_hi = ft_acc.mean() - ft_acc.std(ddof=1), ft_acc.mean() + ft_acc.std(ddof=1)
    ties = []
    for m, rs in data.items():
        if m == "finetune" or any(r.get("diverged", False) for r in rs):
            continue
        a = np.array([r["ACC"] for r in rs]) * 100
        lo, hi = a.mean() - a.std(ddof=1), a.mean() + a.std(ddof=1)
        overlaps = not (hi < ft_lo or lo > ft_hi)
        ties.append(f"{m}:{'TIE' if overlaps else 'DIFF'}")
    rows.append([bench, "TIE-CHECK vs FT (ddof=1 intervals)", "", "; ".join(ties)] + [""] * 7)

os.makedirs("outputs/audit", exist_ok=True)
with open("outputs/audit/recomputed_numbers.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["bench", "method", "n_seeds", "ACC_mean", "ACC_std_ddof0", "ACC_std_ddof1",
                "BWT_mean", "BWT_std_ddof0", "BWT_std_ddof1", "NEW_mean"])
    w.writerows(rows)

for r in rows:
    print(" | ".join(str(x) for x in r))
