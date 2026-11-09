import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cl2

def run_tuning():
    print("=== Step 1: Tuning Repaired Gate Variants (Validation Split, Seeds 0-2) ===")
    
    val_data = cl2.load_mnist(eval_split="val")
    val_seeds = [0, 1, 2]
    
    # 6 configs for drift, 6 configs for hybrid, 4 configs for conflict_soft
    grid = [
        # drift: beta in {0.1, 0.25, 0.5} x decay in {1.0, 0.9}
        ("drift_b01_d10", "gate:drift:0.1", {"beta": 0.1, "decay": 1.0}),
        ("drift_b01_d09", "gate:drift:0.1", {"beta": 0.1, "decay": 0.9}),
        ("drift_b025_d10", "gate:drift:0.25", {"beta": 0.25, "decay": 1.0}),
        ("drift_b025_d09", "gate:drift:0.25", {"beta": 0.25, "decay": 0.9}),
        ("drift_b05_d10", "gate:drift:0.5", {"beta": 0.5, "decay": 1.0}),
        ("drift_b05_d09", "gate:drift:0.5", {"beta": 0.5, "decay": 0.9}),
        
        # hybrid: beta in {0.1, 0.25} x gamma in {0.25, 0.5} x decay in {1.0, 0.9}
        ("hybrid_b01_g025_d10", "gate:hybrid:0.1:0.25", {"beta": 0.1, "gamma": 0.25, "decay": 1.0}),
        ("hybrid_b01_g05_d10", "gate:hybrid:0.1:0.5", {"beta": 0.1, "gamma": 0.5, "decay": 1.0}),
        ("hybrid_b025_g025_d10", "gate:hybrid:0.25:0.25", {"beta": 0.25, "gamma": 0.25, "decay": 1.0}),
        ("hybrid_b025_g05_d10", "gate:hybrid:0.25:0.5", {"beta": 0.25, "gamma": 0.5, "decay": 1.0}),
        ("hybrid_b01_g025_d09", "gate:hybrid:0.1:0.25", {"beta": 0.1, "gamma": 0.25, "decay": 0.9}),
        ("hybrid_b025_g025_d09", "gate:hybrid:0.25:0.25", {"beta": 0.25, "gamma": 0.25, "decay": 0.9}),
        
        # conflict_soft: beta in {0.05, 0.1} x decay in {1.0, 0.9}
        ("confsoft_b005_d10", "gate:conflict_soft:0.05", {"beta": 0.05, "decay": 1.0}),
        ("confsoft_b005_d09", "gate:conflict_soft:0.05", {"beta": 0.05, "decay": 0.9}),
        ("confsoft_b01_d10", "gate:conflict_soft:0.1", {"beta": 0.1, "decay": 1.0}),
        ("confsoft_b01_d09", "gate:conflict_soft:0.1", {"beta": 0.1, "decay": 0.9}),
    ]
    
    os.makedirs("outputs/phase3", exist_ok=True)
    records = []
    
    for tag, method_str, p in grid:
        print(f"Tuning {tag} ({method_str})...", flush=True)
        args = type("Args", (), {
            "tasks": 5, "epochs": 3, "lr": 0.05, "bs": 128, "seeds": 3,
            "seed_start": 0, "eval_split": "val", "bench": "permmnist",
            "out": "", "clip": 0.0, "ewc_lambda": 5000, "buf": 200, "r": 8,
            "beta": p.get("beta", 1.0), "gamma": p.get("gamma", 0.25), "decay": p.get("decay", 1.0)
        })()
        
        runs = [cl2.run(method_str, args, s, val_data) for s in val_seeds]
        
        if any(r.get("diverged", False) for r in runs):
            records.append({
                "tag": tag, "method": method_str, "mode": method_str.split(":")[1], "params": p,
                "ACC_mean": None, "ACC_std": None, "BWT_mean": None, "BWT_std": None,
                "NEW_mean": None, "NEW_std": None, "diverged": True
            })
        else:
            acc = [r["ACC"] * 100 for r in runs]
            bwt = [r["BWT"] * 100 for r in runs]
            fwt = [r["FWT"] * 100 for r in runs]
            new = [r["NEW"] * 100 for r in runs]
            records.append({
                "tag": tag, "method": method_str, "mode": method_str.split(":")[1], "params": p,
                "ACC_mean": float(np.mean(acc)), "ACC_std": float(np.std(acc)),
                "BWT_mean": float(np.mean(bwt)), "BWT_std": float(np.std(bwt)),
                "FWT_mean": float(np.mean(fwt)), "FWT_std": float(np.std(fwt)),
                "NEW_mean": float(np.mean(new)), "NEW_std": float(np.std(new)),
                "diverged": False
            })

    # Save tuning CSV
    csv_rows = []
    for r in records:
        if r["diverged"]:
            csv_rows.append({"tag": r["tag"], "method": r["method"], "params": str(r["params"]), "ACC": "DIVERGED", "BWT": "--", "FWT": "--", "NEW": "--"})
        else:
            csv_rows.append({
                "tag": r["tag"], "method": r["method"], "params": str(r["params"]),
                "ACC": f"{r['ACC_mean']:.2f} ± {r['ACC_std']:.2f}",
                "BWT": f"{r['BWT_mean']:.2f} ± {r['BWT_std']:.2f}",
                "FWT": f"{r['FWT_mean']:.2f} ± {r['FWT_std']:.2f}",
                "NEW": f"{r['NEW_mean']:.2f} ± {r['NEW_std']:.2f}",
            })
    df_tune = pd.DataFrame(csv_rows)
    df_tune.to_csv("outputs/phase3/tuning_repaired_val.csv", index=False)
    print("Saved outputs/phase3/tuning_repaired_val.csv")
    
    # Lock best per repaired mode
    modes = ["drift", "hybrid", "conflict_soft"]
    locked = {}
    for m in modes:
        cand = [r for r in records if not r["diverged"] and r["mode"] == m]
        cand.sort(key=lambda x: (x["ACC_mean"], x["BWT_mean"], x["NEW_mean"]), reverse=True)
        best = cand[0]
        locked[m] = {
            "method": best["method"],
            "tag": best["tag"],
            "ACC": f"{best['ACC_mean']:.2f} ± {best['ACC_std']:.2f}",
            "BWT": f"{best['BWT_mean']:.2f} ± {best['BWT_std']:.2f}",
            "NEW": f"{best['NEW_mean']:.2f} ± {best['NEW_std']:.2f}",
            "params": best["params"]
        }
    with open("configs/locked_repaired.json", "w") as f:
        json.dump(locked, f, indent=2)
    print("Saved configs/locked_repaired.json:")
    print(json.dumps(locked, indent=2))
    return locked

def run_evaluation(locked):
    print("\n=== Step 2: Final Evaluation of Repaired Variants (5 Seeds 100-104, Test Split) ===")
    
    benchmarks = [
        ("perm5", "permmnist", 5),
        ("perm10", "permmnist", 10),
        ("splitmnist", "splitmnist", 5),
    ]
    
    test_seeds = [100, 101, 102, 103, 104]
    test_data = cl2.load_mnist(eval_split="test")
    
    for b_name, b_type, n_tasks in benchmarks:
        print(f"\n--- Benchmark: {b_name} ({b_type}, {n_tasks} tasks) ---")
        
        # We include finetune as reference
        b_methods = [("finetune", "finetune", {})]
        for m, info in locked.items():
            b_methods.append((f"gate:{m}", info["method"], info["params"]))
            
        b_results = {}
        for label, m_str, p in b_methods:
            print(f"Running {label} ({m_str})...", flush=True)
            args = type("Args", (), {
                "tasks": n_tasks, "epochs": 3, "lr": 0.05, "bs": 128, "seeds": 5,
                "seed_start": 100, "eval_split": "test", "bench": b_type,
                "out": "", "clip": 0.0, "ewc_lambda": 5000, "buf": 200, "r": 8,
                "beta": p.get("beta", 1.0), "gamma": p.get("gamma", 0.25), "decay": p.get("decay", 1.0)
            })()
            runs = [cl2.run(m_str, args, s, test_data) for s in test_seeds]
            b_results[label] = runs
            
        out_json = f"outputs/phase3/final_repaired_{b_name}.json"
        with open(out_json, "w") as f:
            json.dump(b_results, f, indent=2)
        print(f"Saved {out_json}")
        
        # Paired differences vs finetune
        ft_acc = np.array([r["ACC"] * 100 for r in b_results["finetune"]])
        ft_bwt = np.array([r["BWT"] * 100 for r in b_results["finetune"]])
        
        rows = []
        for label, runs in b_results.items():
            acc = np.array([r["ACC"] * 100 for r in runs])
            bwt = np.array([r["BWT"] * 100 for r in runs])
            fwt = np.array([r["FWT"] * 100 for r in runs])
            new = np.array([r["NEW"] * 100 for r in runs])
            diff_acc = acc - ft_acc
            diff_bwt = bwt - ft_bwt
            
            rows.append({
                "Benchmark": b_name,
                "Method": label,
                "ACC": f"{acc.mean():.2f} ± {acc.std():.2f}",
                "Δ ACC vs FT": f"{diff_acc.mean():+.2f} ± {diff_acc.std():.2f}" if label != "finetune" else "reference",
                "BWT": f"{bwt.mean():.2f} ± {bwt.std():.2f}",
                "Δ BWT vs FT": f"{diff_bwt.mean():+.2f} ± {diff_bwt.std():.2f}" if label != "finetune" else "reference",
                "FWT": f"{fwt.mean():.2f} ± {fwt.std():.2f}",
                "NEW": f"{new.mean():.2f} ± {new.std():.2f}",
            })
            
        df_res = pd.DataFrame(rows)
        csv_out = f"outputs/phase3/final_repaired_{b_name}_table.csv"
        df_res.to_csv(csv_out, index=False)
        print(f"Saved {csv_out}")
        print("\n" + df_res.to_string())

def run_layer_analysis(locked):
    print("\n=== Task 3.4: Layer Analysis for Best Gate Variant ===")
    
    # Pick the best gate variant across benchmarks
    # We will test on Permuted-MNIST 5 tasks
    best_m_str = locked["drift"]["method"]
    p = locked["drift"]["params"]
    print(f"Analyzing layer-wise gated fraction for {best_m_str}...")
    
    seeds = [100, 101, 102]
    tasks = 5
    epochs = 3
    bs = 128
    lr = 0.05
    data = cl2.load_mnist(eval_split="test")
    
    # Hook into Gate to record per-layer gated fraction
    layer_names = ["l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]
    # Matrix: rows = layers, columns = tasks 2..5
    task_layer_gated = {name: [[] for _ in range(tasks)] for name in layer_names}
    
    beta = p.get("beta", 0.1)
    decay = p.get("decay", 1.0)
    
    for s in seeds:
        cl2.set_seed(s)
        Xtr, ytr, Xte, yte = data
        perms = cl2.make_perms(tasks, s)
        model = cl2.MLP(n_tasks=tasks, lora=False).to(cl2.dev)
        gate = cl2.Gate(model, "drift", beta, decay)
        
        for t, perm in enumerate(perms):
            Xp = Xtr[:, perm]
            gate.begin_task(model)
            n = Xp.size(0)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            
            for ep in range(epochs):
                idx = torch.randperm(n, device=cl2.dev)
                for i in range(0, n, bs):
                    b = idx[i:i + bs]
                    loss = F.cross_entropy(model(Xp[b]), ytr[b])
                    opt.zero_grad()
                    loss.backward()
                    
                    # Track per-layer gated fraction
                    for (name, param), d, s_weight in zip(model.named_parameters(), gate.d, gate.start):
                        if param.grad is None or not d.any():
                            continue
                        mag = (d.abs() / (d.abs().mean() + 1e-12)).clamp(max=1.0)
                        e = param.detach() - s_weight
                        cond = ((param.grad * d > 0) | (param.grad * e < 0)).float()
                        scale = 1.0 - beta * mag * cond
                        g_frac = (scale < 0.99999).float().mean().item()
                        task_layer_gated[name][t].append(g_frac)
                            
                    gate.apply(model)
                    opt.step()
            gate.end_task(model)
            
    # Compute mean matrix for tasks 2 to 5 (index 1 to 4)
    cols = [f"Task {t+1}" for t in range(1, tasks)]
    heatmap_matrix = np.zeros((len(layer_names), len(cols)))
    
    for i, name in enumerate(layer_names):
        for j, t in enumerate(range(1, tasks)):
            heatmap_matrix[i, j] = np.mean(task_layer_gated[name][t]) * 100
            
    # Plot Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    im = ax.imshow(heatmap_matrix, cmap="Blues", vmin=0, vmax=max(50, heatmap_matrix.max() * 1.1))
    
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(layer_names)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticklabels(layer_names, fontsize=10)
    ax.set_xlabel("Task Learned", fontsize=11)
    ax.set_ylabel("Layer Parameter", fontsize=11)
    ax.set_title(f"Average Gated Fraction (%): {best_m_str}", fontsize=12, fontweight="bold")
    
    for i in range(len(layer_names)):
        for j in range(len(cols)):
            val = heatmap_matrix[i, j]
            color = "white" if val > 25 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
            
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png = "outputs/phase3/layer_heatmap_best_gate.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")
    
    # Print measured layer gating ranking
    mean_per_layer = heatmap_matrix.mean(axis=1)
    ranking = sorted(zip(layer_names, mean_per_layer), key=lambda x: x[1], reverse=True)
    print("\nLayer Gating Hierarchy (MEASURED):")
    for name, val in ranking:
        print(f"  {name:12s}: {val:.2f}%")

if __name__ == "__main__":
    if os.path.exists("configs/locked_repaired.json") and os.path.exists("outputs/phase3/final_repaired_splitmnist_table.csv"):
        print("Tuning and evaluation files exist. Proceeding directly to layer analysis...")
        with open("configs/locked_repaired.json") as f:
            locked = json.load(f)
        run_layer_analysis(locked)
    else:
        locked = run_tuning()
        run_evaluation(locked)
        run_layer_analysis(locked)
