import os, sys, json, subprocess, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cl2

def run_tuning():
    print("=== Step 1: Tuning Grid on Split-MNIST (Validation Split, Seeds 0-2) ===")
    
    val_data = cl2.load_mnist(eval_split="val")
    val_seeds = [0, 1, 2]
    
    grid = [
        # finetune
        ("finetune", "finetune", {}),
        # ewc
        ("ewc_lam1", "ewc", {"ewc_lambda": 1.0}),
        ("ewc_lam5", "ewc", {"ewc_lambda": 5.0}),
        ("ewc_lam20", "ewc", {"ewc_lambda": 20.0}),
        ("ewc_lam100", "ewc", {"ewc_lambda": 100.0}),
        # replay
        ("replay_buf50", "replay", {"buf": 50}),
        ("replay_buf200", "replay", {"buf": 200}),
        ("replay_buf1000", "replay", {"buf": 1000}),
        # lora
        ("lora_r4", "lora", {"r": 4}),
        ("lora_r8", "lora", {"r": 8}),
        ("lora_r16", "lora", {"r": 16}),
        # gate:conflict
        ("gate_conf_b025_d10", "gate:conflict:0.25", {"decay": 1.0}),
        ("gate_conf_b025_d09", "gate:conflict:0.25", {"decay": 0.9}),
        ("gate_conf_b05_d10", "gate:conflict:0.5", {"decay": 1.0}),
        ("gate_conf_b05_d09", "gate:conflict:0.5", {"decay": 0.9}),
        ("gate_conf_b10_d10", "gate:conflict:1.0", {"decay": 1.0}),
        ("gate_conf_b10_d09", "gate:conflict:1.0", {"decay": 0.9}),
        # gate:magnitude
        ("gate_mag_b025_d10", "gate:magnitude:0.25", {"decay": 1.0}),
        ("gate_mag_b025_d09", "gate:magnitude:0.25", {"decay": 0.9}),
        ("gate_mag_b05_d10", "gate:magnitude:0.5", {"decay": 1.0}),
        ("gate_mag_b05_d09", "gate:magnitude:0.5", {"decay": 0.9}),
        ("gate_mag_b10_d10", "gate:magnitude:1.0", {"decay": 1.0}),
        ("gate_mag_b10_d09", "gate:magnitude:1.0", {"decay": 0.9}),
        # gate:layer_cos
        ("gate_cos_b05_d10", "gate:layer_cos:0.5", {"decay": 1.0}),
        ("gate_cos_b10_d10", "gate:layer_cos:1.0", {"decay": 1.0}),
    ]
    
    os.makedirs("outputs/phase2", exist_ok=True)
    tuning_records = []
    
    for tag, method_str, p in grid:
        print(f"Tuning {tag} ({method_str})...", flush=True)
        args = type("Args", (), {
            "tasks": 5, "epochs": 3, "lr": 0.05, "bs": 128, "seeds": 3,
            "seed_start": 0, "eval_split": "val", "bench": "splitmnist",
            "out": f"phase2/tune_split_{tag}", "clip": 0.0,
            "ewc_lambda": p.get("ewc_lambda", 5000),
            "buf": p.get("buf", 200),
            "r": p.get("r", 8),
            "beta": 1.0,
            "decay": p.get("decay", 1.0)
        })()
        
        runs = [cl2.run(method_str, args, s, val_data) for s in val_seeds]
        
        # Save individual run file
        with open(f"outputs/phase2/results_tune_split_{tag}.json", "w") as f:
            json.dump({method_str: runs}, f, indent=2)
            
        if any(r.get("diverged", False) for r in runs):
            tuning_records.append({
                "tag": tag, "method": method_str, "params": p,
                "ACC_mean": None, "ACC_std": None,
                "BWT_mean": None, "BWT_std": None,
                "NEW_mean": None, "NEW_std": None,
                "diverged": True
            })
        else:
            acc = [r["ACC"] * 100 for r in runs]
            bwt = [r["BWT"] * 100 for r in runs]
            fwt = [r["FWT"] * 100 for r in runs]
            new = [r["NEW"] * 100 for r in runs]
            tuning_records.append({
                "tag": tag, "method": method_str, "params": p,
                "ACC_mean": float(np.mean(acc)), "ACC_std": float(np.std(acc)),
                "BWT_mean": float(np.mean(bwt)), "BWT_std": float(np.std(bwt)),
                "FWT_mean": float(np.mean(fwt)), "FWT_std": float(np.std(fwt)),
                "NEW_mean": float(np.mean(new)), "NEW_std": float(np.std(new)),
                "diverged": False
            })

    # Save tuning CSV
    csv_rows = []
    for r in tuning_records:
        if r["diverged"]:
            csv_rows.append({
                "tag": r["tag"], "method": r["method"], "params": str(r["params"]),
                "ACC": "DIVERGED", "BWT": "--", "FWT": "--", "NEW": "--"
            })
        else:
            csv_rows.append({
                "tag": r["tag"], "method": r["method"], "params": str(r["params"]),
                "ACC": f"{r['ACC_mean']:.2f} ± {r['ACC_std']:.2f}",
                "BWT": f"{r['BWT_mean']:.2f} ± {r['BWT_std']:.2f}",
                "FWT": f"{r['FWT_mean']:.2f} ± {r['FWT_std']:.2f}",
                "NEW": f"{r['NEW_mean']:.2f} ± {r['NEW_std']:.2f}",
            })
            
    df_tune = pd.DataFrame(csv_rows)
    df_tune.to_csv("outputs/phase2/tuning_split_val.csv", index=False)
    print("Saved outputs/phase2/tuning_split_val.csv")
    
    # Lock best configurations
    families = ["finetune", "ewc", "replay", "lora", "gate:conflict", "gate:magnitude", "gate:layer_cos"]
    locked = {}
    for fam in families:
        cand = [r for r in tuning_records if not r["diverged"] and r["method"].startswith(fam)]
        cand.sort(key=lambda x: (x["ACC_mean"], x["BWT_mean"], x["NEW_mean"]), reverse=True)
        best = cand[0]
        locked[fam] = {
            "method": best["method"],
            "tag": best["tag"],
            "ACC": f"{best['ACC_mean']:.2f} ± {best['ACC_std']:.2f}",
            "BWT": f"{best['BWT_mean']:.2f} ± {best['BWT_std']:.2f}",
            "NEW": f"{best['NEW_mean']:.2f} ± {best['NEW_std']:.2f}",
            "params": best["params"]
        }
        
    with open("configs/locked_split.json", "w") as f:
        json.dump(locked, f, indent=2)
    print("Saved configs/locked_split.json:")
    print(json.dumps(locked, indent=2))
    return locked

def run_final(locked):
    print("\n=== Step 2: Final Run on Split-MNIST (Test Split, Seeds 100-104) ===")
    test_data = cl2.load_mnist(eval_split="test")
    test_seeds = [100, 101, 102, 103, 104]
    
    final_results = {}
    for fam, info in locked.items():
        m_str = info["method"]
        p = info["params"]
        print(f"\nRunning final {fam} ({m_str})...", flush=True)
        args = type("Args", (), {
            "tasks": 5, "epochs": 3, "lr": 0.05, "bs": 128, "seeds": 5,
            "seed_start": 100, "eval_split": "test", "bench": "splitmnist",
            "out": "", "clip": 0.0,
            "ewc_lambda": p.get("ewc_lambda", 5000),
            "buf": p.get("buf", 200),
            "r": p.get("r", 8),
            "beta": 1.0,
            "decay": p.get("decay", 1.0)
        })()
        
        runs = []
        for s in test_seeds:
            res = cl2.run(m_str, args, s, test_data)
            runs.append(res)
            print(f"  Seed {s}: ACC={res['ACC']*100:.2f}%, BWT={res['BWT']*100:.2f}%, NEW={res['NEW']*100:.2f}%", flush=True)
        final_results[fam] = runs

    # Save final JSON
    out_json = "outputs/phase2/final_split.json"
    with open(out_json, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved {out_json}")
    
    # Paired differences vs finetune
    ft_runs = final_results["finetune"]
    ft_acc = np.array([r["ACC"] * 100 for r in ft_runs])
    ft_bwt = np.array([r["BWT"] * 100 for r in ft_runs])
    
    table_rows = []
    print(f"\n{'Method':<18}{'ACC (%)':>16}{'diff vs FT':>16}{'BWT (%)':>16}{'diff vs FT':>16}{'FWT (%)':>16}{'NEW (%)':>16}")
    print("-" * 102)
    
    for fam in final_results:
        runs = final_results[fam]
        acc = np.array([r["ACC"] * 100 for r in runs])
        bwt = np.array([r["BWT"] * 100 for r in runs])
        fwt = np.array([r["FWT"] * 100 for r in runs])
        new = np.array([r["NEW"] * 100 for r in runs])
        
        diff_acc = acc - ft_acc
        diff_bwt = bwt - ft_bwt
        
        acc_s = f"{acc.mean():.2f} ± {acc.std():.2f}"
        d_acc_s = f"{diff_acc.mean():+.2f} ± {diff_acc.std():.2f}" if fam != "finetune" else "reference"
        bwt_s = f"{bwt.mean():.2f} ± {bwt.std():.2f}"
        d_bwt_s = f"{diff_bwt.mean():+.2f} ± {diff_bwt.std():.2f}" if fam != "finetune" else "reference"
        fwt_s = f"{fwt.mean():.2f} ± {fwt.std():.2f}"
        new_s = f"{new.mean():.2f} ± {new.std():.2f}"
        
        print(f"{fam:<18}{acc_s:>16}{d_acc_s:>16}{bwt_s:>16}{d_bwt_s:>16}{fwt_s:>16}{new_s:>16}")
        
        table_rows.append({
            "Method": fam,
            "ACC": acc_s,
            "ACC_diff_vs_FT": d_acc_s,
            "BWT": bwt_s,
            "BWT_diff_vs_FT": d_bwt_s,
            "FWT": fwt_s,
            "NEW": new_s,
            "ACC_mean": acc.mean(), "ACC_std": acc.std(),
            "BWT_mean": bwt.mean(), "BWT_std": bwt.std(),
        })

    df = pd.DataFrame(table_rows)
    csv_path = "outputs/phase2/final_split_table.csv"
    df[["Method", "ACC", "ACC_diff_vs_FT", "BWT", "BWT_diff_vs_FT", "FWT", "NEW"]].to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    # Table image
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    col_labels = ["Method", "ACC (%)", "Δ ACC vs FT", "BWT (%)", "Δ BWT vs FT", "FWT (%)", "NEW (%)"]
    cell_text = df[["Method", "ACC", "ACC_diff_vs_FT", "BWT", "BWT_diff_vs_FT", "FWT", "NEW"]].values
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.5)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(weight="bold")
    best_acc_idx = np.argmax(df["ACC_mean"])
    best_bwt_idx = np.argmax(df["BWT_mean"])
    tbl[(best_acc_idx + 1, 1)].set_text_props(weight="bold", color="darkblue")
    tbl[(best_bwt_idx + 1, 3)].set_text_props(weight="bold", color="darkgreen")
    
    plt.tight_layout()
    tbl_img = "outputs/phase2/final_split_table.png"
    plt.savefig(tbl_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {tbl_img}")
    
    # Forgetting Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tasks = np.arange(1, 6)
    colors = {
        "finetune": "#1f77b4", "ewc": "#ff7f0e", "replay": "#2ca02c",
        "lora": "#d62728", "gate:conflict": "#9467bd",
        "gate:magnitude": "#8c564b", "gate:layer_cos": "#e377c2"
    }
    
    for fam in final_results:
        runs = final_results[fam]
        r0 = np.array([np.array(r["R"])[:, 0] * 100 for r in runs])
        m = r0.mean(axis=0)
        s = r0.std(axis=0)
        c = colors.get(fam, "black")
        ax.plot(tasks, m, marker="o", label=f"{fam} (final {m[-1]:.1f}%)", color=c, linewidth=1.8)
        ax.fill_between(tasks, m - s, m + s, alpha=0.15, color=c)
        
    ax.set_xlabel("Tasks Learned So Far", fontsize=11)
    ax.set_ylabel("Accuracy on Task 1 (Digits 0/1) (%)", fontsize=11)
    ax.set_title("Task 1 Forgetting Curve (Split-MNIST 5 tasks, 5 seeds, Test set)", fontsize=12, fontweight="bold")
    ax.set_xticks(tasks)
    ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    curve_img = "outputs/phase2/forgetting_curve_split.png"
    plt.savefig(curve_img, dpi=200)
    plt.close()
    print(f"Saved {curve_img}")
    
    # Heatmaps per method
    n_m = len(final_results)
    cols = 3
    rows = int(np.ceil(n_m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8), squeeze=False)
    
    for idx, (fam, runs) in enumerate(final_results.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        R_mean = np.mean([np.array(run["R"]) for run in runs], axis=0) * 100
        im = ax.imshow(R_mean, cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_title(fam, fontsize=11, fontweight="bold")
        ax.set_xlabel("Task j", fontsize=9)
        ax.set_ylabel("After task i", fontsize=9)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"T{j+1}" for j in range(5)])
        ax.set_yticklabels([f"T{i+1}" for i in range(5)])
        for i in range(5):
            for j in range(5):
                val = R_mean[i, j]
                color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)
                
    for idx in range(n_m, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
        
    fig.suptitle("Accuracy Matrix R (%) — Split-MNIST (5 tasks, 5 seeds, Test set)", fontsize=13, y=1.01)
    fig.tight_layout()
    hm_img = "outputs/phase2/heatmaps_split.png"
    plt.savefig(hm_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {hm_img}")

if __name__ == "__main__":
    locked = run_tuning()
    run_final(locked)
