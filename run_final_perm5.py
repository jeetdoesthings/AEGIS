import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cl2

def main():
    print("=== Task 1.4: Final Run on Permuted-MNIST (5 tasks, 5 seeds 100-104, test split) ===")
    
    seeds = [100, 101, 102, 103, 104]
    data = cl2.load_mnist(eval_split="test")
    
    # Locked configurations from configs/locked_perm5.json
    with open("configs/locked_perm5.json") as f:
        locked = json.load(f)
        
    print("Locked configurations loaded:")
    for k, v in locked.items():
        print(f"  {k}: {v['method']} (params: {v['params']})")
        
    methods_to_run = [
        ("finetune", "finetune", {"ewc_lambda": 5000, "buf": 200, "r": 8, "decay": 1.0}),
        ("ewc", "ewc", {"ewc_lambda": 20.0, "buf": 200, "r": 8, "decay": 1.0}),
        ("replay", "replay", {"ewc_lambda": 5000, "buf": 1000, "r": 8, "decay": 1.0}),
        ("lora", "lora", {"ewc_lambda": 5000, "buf": 200, "r": 16, "decay": 1.0}),
        ("gate:conflict", "gate:conflict:0.25", {"ewc_lambda": 5000, "buf": 200, "r": 8, "decay": 1.0}),
        ("gate:magnitude", "gate:magnitude:1.0", {"ewc_lambda": 5000, "buf": 200, "r": 8, "decay": 1.0}),
        ("gate:layer_cos", "gate:layer_cos:1.0", {"ewc_lambda": 5000, "buf": 200, "r": 8, "decay": 1.0}),
    ]
    
    all_results = {}
    
    for label, m_str, p in methods_to_run:
        print(f"\nRunning {label} ({m_str})...", flush=True)
        args = type("Args", (), {
            "tasks": 5, "epochs": 3, "lr": 0.05, "bs": 128, "seeds": len(seeds),
            "seed_start": 100, "eval_split": "test", "out": "", "clip": 0.0,
            "ewc_lambda": p["ewc_lambda"], "buf": p["buf"], "r": p["r"],
            "beta": 1.0, "decay": p["decay"]
        })()
        
        runs = []
        for s in seeds:
            res = cl2.run(m_str, args, s, data)
            runs.append(res)
            print(f"  Seed {s}: ACC={res['ACC']*100:.2f}%, BWT={res['BWT']*100:.2f}%, NEW={res['NEW']*100:.2f}%", flush=True)
        all_results[label] = runs

    # Save outputs/phase1/final_perm5.json
    out_json = "outputs/phase1/final_perm5.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results to {out_json}")
    
    # Paired differences vs finetune per seed
    ft_runs = all_results["finetune"]
    ft_acc = np.array([r["ACC"] * 100 for r in ft_runs])
    ft_bwt = np.array([r["BWT"] * 100 for r in ft_runs])
    
    table_rows = []
    print(f"\n{'Method':<18}{'ACC (%)':>16}{'diff vs FT':>16}{'BWT (%)':>16}{'diff vs FT':>16}{'FWT (%)':>16}{'NEW (%)':>16}")
    print("-" * 102)
    
    for label in all_results:
        runs = all_results[label]
        acc = np.array([r["ACC"] * 100 for r in runs])
        bwt = np.array([r["BWT"] * 100 for r in runs])
        fwt = np.array([r["FWT"] * 100 for r in runs])
        new = np.array([r["NEW"] * 100 for r in runs])
        
        diff_acc = acc - ft_acc
        diff_bwt = bwt - ft_bwt
        
        acc_s = f"{acc.mean():.2f} ± {acc.std():.2f}"
        d_acc_s = f"{diff_acc.mean():+.2f} ± {diff_acc.std():.2f}" if label != "finetune" else "reference"
        bwt_s = f"{bwt.mean():.2f} ± {bwt.std():.2f}"
        d_bwt_s = f"{diff_bwt.mean():+.2f} ± {diff_bwt.std():.2f}" if label != "finetune" else "reference"
        fwt_s = f"{fwt.mean():.2f} ± {fwt.std():.2f}"
        new_s = f"{new.mean():.2f} ± {new.std():.2f}"
        
        print(f"{label:<18}{acc_s:>16}{d_acc_s:>16}{bwt_s:>16}{d_bwt_s:>16}{fwt_s:>16}{new_s:>16}")
        
        table_rows.append({
            "Method": label,
            "ACC": acc_s,
            "ACC_diff_vs_FT": d_acc_s,
            "BWT": bwt_s,
            "BWT_diff_vs_FT": d_bwt_s,
            "FWT": fwt_s,
            "NEW": new_s,
            "ACC_mean": acc.mean(), "ACC_std": acc.std(),
            "BWT_mean": bwt.mean(), "BWT_std": bwt.std(),
        })

    # Save CSV
    df = pd.DataFrame(table_rows)
    csv_path = "outputs/phase1/final_perm5_table.csv"
    df[["Method", "ACC", "ACC_diff_vs_FT", "BWT", "BWT_diff_vs_FT", "FWT", "NEW"]].to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    # Save Table Image
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
    # Highlight best ACC and BWT
    best_acc_idx = np.argmax(df["ACC_mean"])
    best_bwt_idx = np.argmax(df["BWT_mean"])
    tbl[(best_acc_idx + 1, 1)].set_text_props(weight="bold", color="darkblue")
    tbl[(best_bwt_idx + 1, 3)].set_text_props(weight="bold", color="darkgreen")
    
    plt.tight_layout()
    tbl_img = "outputs/phase1/final_perm5_table.png"
    plt.savefig(tbl_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {tbl_img}")
    
    # Plot Forgetting curve with mean +- std band
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tasks = np.arange(1, 6)
    
    colors = {
        "finetune": "#1f77b4", "ewc": "#ff7f0e", "replay": "#2ca02c",
        "lora": "#d62728", "gate:conflict": "#9467bd",
        "gate:magnitude": "#8c564b", "gate:layer_cos": "#e377c2"
    }
    
    for label in all_results:
        runs = all_results[label]
        # R is 5x5. Task 1 accuracy after task i is R[i, 0]
        r0 = np.array([np.array(r["R"])[:, 0] * 100 for r in runs])
        m = r0.mean(axis=0)
        s = r0.std(axis=0)
        c = colors.get(label, "black")
        ax.plot(tasks, m, marker="o", label=f"{label} (final {m[-1]:.1f}%)", color=c, linewidth=1.8)
        ax.fill_between(tasks, m - s, m + s, alpha=0.15, color=c)
        
    ax.set_xlabel("Tasks Learned So Far", fontsize=11)
    ax.set_ylabel("Accuracy on Task 1 (%)", fontsize=11)
    ax.set_title("Task 1 Forgetting Curve (Permuted-MNIST 5 tasks, 5 seeds, Test set)", fontsize=12, fontweight="bold")
    ax.set_xticks(tasks)
    ax.set_xticklabels([f"Task {t}" for t in tasks])
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    curve_img = "outputs/phase1/forgetting_curve_perm5.png"
    plt.savefig(curve_img, dpi=200)
    plt.close()
    print(f"Saved {curve_img}")
    
    # Accuracy-matrix heatmaps per method in a grid
    n_m = len(all_results)
    cols = 3
    rows = int(np.ceil(n_m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8), squeeze=False)
    
    for idx, (label, runs) in enumerate(all_results.items()):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        R_mean = np.mean([np.array(run["R"]) for run in runs], axis=0) * 100
        im = ax.imshow(R_mean, cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Task j", fontsize=9)
        ax.set_ylabel("After learning task i", fontsize=9)
        ax.set_xticks(range(R_mean.shape[1]))
        ax.set_yticks(range(R_mean.shape[0]))
        ax.set_xticklabels([f"T{j+1}" for j in range(R_mean.shape[1])])
        ax.set_yticklabels([f"T{i+1}" for i in range(R_mean.shape[0])])
        for i in range(R_mean.shape[0]):
            for j in range(R_mean.shape[1]):
                val = R_mean[i, j]
                color = "white" if val > 65 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)
                
    for idx in range(n_m, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
        
    fig.suptitle("Accuracy Matrix R (%) — Permuted-MNIST (5 tasks, 5 seeds, Test set)", fontsize=13, y=1.01)
    fig.tight_layout()
    hm_img = "outputs/phase1/heatmaps_perm5.png"
    plt.savefig(hm_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {hm_img}")

if __name__ == "__main__":
    main()
