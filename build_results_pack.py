import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS_ORDER = [
    "finetune",
    "ewc",
    "replay",
    "lora",
    "gate:conflict",
    "gate:magnitude",
    "gate:layer_cos",
    "gate:drift",
    "gate:hybrid",
    "gate:conflict_soft",
]

METHOD_NAMES_PRETTY = {
    "finetune": "Fine-Tuning (Plain SGD)",
    "ewc": "EWC (repaired, λ=20/100)",
    "replay": "Replay (buffer=1000)",
    "lora": "LoRA (r=16/8, oracle task ID)*",
    "gate:conflict": "Gate: Conflict (original, β=0.25)",
    "gate:magnitude": "Gate: Magnitude (ablation, β=1.0)",
    "gate:layer_cos": "Gate: Layer Cosine (β=0.5/1.0)",
    "gate:drift": "Gate: Drift (anti-drift gate, β=0.1)",
    "gate:hybrid": "Gate: Hybrid (conflict × mag, β=0.1)",
    "gate:conflict_soft": "Gate: Soft Conflict (β=0.1)",
}

def load_combined_results(file_base, file_repaired):
    with open(file_base) as f:
        d_base = json.load(f)
    with open(file_repaired) as f:
        d_rep = json.load(f)
    
    combined = {}
    for m in METHODS_ORDER:
        if m in d_rep and m != "finetune":
            combined[m] = d_rep[m]
        elif m in d_base:
            combined[m] = d_base[m]
    return combined

def compute_table_df(combined):
    ft_runs = combined["finetune"]
    ft_acc = np.array([r["ACC"] for r in ft_runs]) * 100.0
    ft_bwt = np.array([r["BWT"] for r in ft_runs]) * 100.0
    
    rows = []
    for m in METHODS_ORDER:
        if m not in combined:
            continue
        runs = combined[m]
        acc = np.array([r["ACC"] for r in runs]) * 100.0
        bwt = np.array([r["BWT"] for r in runs]) * 100.0
        fwt = np.array([r["FWT"] for r in runs]) * 100.0
        new = np.array([r["NEW"] for r in runs]) * 100.0
        
        if m == "finetune":
            d_acc_str = "reference"
            d_bwt_str = "reference"
        else:
            d_acc = acc - ft_acc
            d_bwt = bwt - ft_bwt
            d_acc_str = f"{d_acc.mean():+.2f} ± {d_acc.std():.2f}"
            d_bwt_str = f"{d_bwt.mean():+.2f} ± {d_bwt.std():.2f}"
            
        rows.append({
            "Method": m,
            "Method Display": METHOD_NAMES_PRETTY.get(m, m),
            "ACC (%)": f"{acc.mean():.2f} ± {acc.std():.2f}",
            "Δ ACC vs FT": d_acc_str,
            "BWT (%)": f"{bwt.mean():.2f} ± {bwt.std():.2f}",
            "Δ BWT vs FT": d_bwt_str,
            "FWT (%)": f"{fwt.mean():.2f} ± {fwt.std():.2f}",
            "NEW (%)": f"{new.mean():.2f} ± {new.std():.2f}",
            "ACC_mean": acc.mean(),
            "ACC_std": acc.std(),
            "BWT_mean": bwt.mean(),
            "BWT_std": bwt.std(),
        })
    return pd.DataFrame(rows)

def render_table_image(df, title, out_png):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    col_labels = ["Method", "ACC (%)", "Δ ACC vs FT", "BWT (%)", "Δ BWT vs FT", "FWT (%)", "NEW (%)"]
    cell_text = df[["Method Display", "ACC (%)", "Δ ACC vs FT", "BWT (%)", "Δ BWT vs FT", "FWT (%)", "NEW (%)"]].values
    
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.15, 1.45)
    
    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(weight="bold", color="black")
        tbl[(0, j)].set_facecolor("#e6e6e6")
        
    best_acc_idx = np.argmax(df["ACC_mean"])
    best_bwt_idx = np.argmax(df["BWT_mean"])
    tbl[(best_acc_idx + 1, 1)].set_text_props(weight="bold", color="darkblue")
    tbl[(best_bwt_idx + 1, 3)].set_text_props(weight="bold", color="darkgreen")
    
    ax.set_title(title, fontsize=12, pad=15, weight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")

def plot_comprehensive_forgetting(combined, num_tasks, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    tasks = np.arange(1, num_tasks + 1)
    
    palette = {
        "finetune": ("#1f77b4", "-"),
        "ewc": ("#ff7f0e", "-"),
        "replay": ("#2ca02c", "-"),
        "lora": ("#d62728", "--"),
        "gate:conflict": ("#9467bd", "-."),
        "gate:magnitude": ("#8c564b", "-"),
        "gate:drift": ("#17becf", "-"),
        "gate:hybrid": ("#bcbd22", "-"),
        "gate:conflict_soft": ("#7f7f7f", ":"),
    }
    
    for m in METHODS_ORDER:
        if m not in combined or m == "gate:layer_cos":
            continue
        color, ls = palette.get(m, ("black", "-"))
        runs = combined[m]
        r1_all = []
        for r in runs:
            R = np.array(r["R"])
            r1_all.append(R[:, 0])
        r1_all = np.array(r1_all) * 100.0
        mean = r1_all.mean(axis=0)
        std = r1_all.std(axis=0)
        
        lbl = m
        if m == "lora":
            lbl += " (oracle ID)*"
        ax.plot(tasks, mean, label=lbl, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(tasks, mean - std, mean + std, color=color, alpha=0.1)
        
    ax.set_xlabel("After Learning Task $i$", fontsize=11)
    ax.set_ylabel("Accuracy on Task 1 (%)", fontsize=11)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks(tasks)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")

def main():
    print("Building Results Pack for Phase 5...")
    
    # 1. Perm5
    comb_perm5 = load_combined_results("outputs/phase1/final_perm5.json", "outputs/phase3/final_repaired_perm5.json")
    df_perm5 = compute_table_df(comb_perm5)
    df_perm5[["Method", "Method Display", "ACC (%)", "Δ ACC vs FT", "BWT (%)", "Δ BWT vs FT", "FWT (%)", "NEW (%)"]].to_csv("outputs/phase5/table_perm5.csv", index=False)
    render_table_image(df_perm5, "Permuted-MNIST 5 Tasks Evaluation (5 Seeds: 100-104, Held-Out Test Split)", "outputs/phase5/table_perm5.png")
    plot_comprehensive_forgetting(comb_perm5, 5, "Permuted-MNIST 5-Task: Task 1 Accuracy Retention Across Tasks (Mean ± Std, 5 Seeds)", "outputs/phase5/forgetting_curve_perm5_all.png")

    # 2. Perm10
    comb_perm10 = load_combined_results("outputs/phase2/final_perm10.json", "outputs/phase3/final_repaired_perm10.json")
    df_perm10 = compute_table_df(comb_perm10)
    df_perm10[["Method", "Method Display", "ACC (%)", "Δ ACC vs FT", "BWT (%)", "Δ BWT vs FT", "FWT (%)", "NEW (%)"]].to_csv("outputs/phase5/table_perm10.csv", index=False)
    render_table_image(df_perm10, "Permuted-MNIST 10 Tasks Evaluation (5 Seeds: 100-104, Held-Out Test Split)", "outputs/phase5/table_perm10.png")
    plot_comprehensive_forgetting(comb_perm10, 10, "Permuted-MNIST 10-Task: Task 1 Accuracy Retention Across Tasks (Mean ± Std, 5 Seeds)", "outputs/phase5/forgetting_curve_perm10_all.png")

    # 3. Split-MNIST
    comb_split = load_combined_results("outputs/phase2/final_split.json", "outputs/phase3/final_repaired_splitmnist.json")
    df_split = compute_table_df(comb_split)
    df_split[["Method", "Method Display", "ACC (%)", "Δ ACC vs FT", "BWT (%)", "Δ BWT vs FT", "FWT (%)", "NEW (%)"]].to_csv("outputs/phase5/table_split.csv", index=False)
    render_table_image(df_split, "Split-MNIST Domain-Incremental 5 Tasks Evaluation (5 Seeds: 100-104, Held-Out Test Split)", "outputs/phase5/table_split.png")

    # 4. Master Cross-Benchmark Summary Table
    master_rows = []
    for m in METHODS_ORDER:
        r_p5 = df_perm5[df_perm5["Method"] == m].iloc[0]
        r_p10 = df_perm10[df_perm10["Method"] == m].iloc[0]
        r_sp = df_split[df_split["Method"] == m].iloc[0]
        master_rows.append({
            "Method": m,
            "Method Display": METHOD_NAMES_PRETTY.get(m, m),
            "Perm5 ACC": r_p5["ACC (%)"],
            "Perm5 BWT": r_p5["BWT (%)"],
            "Perm10 ACC": r_p10["ACC (%)"],
            "Perm10 BWT": r_p10["BWT (%)"],
            "Split ACC": r_sp["ACC (%)"],
            "Split BWT": r_sp["BWT (%)"],
        })
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv("outputs/phase5/master_summary.csv", index=False)
    
    # Render Master Table Image
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    col_labels = ["Method", "Perm5 ACC (%)", "Perm5 BWT (%)", "Perm10 ACC (%)", "Perm10 BWT (%)", "Split ACC (%)", "Split BWT (%)"]
    cell_text = df_master[["Method Display", "Perm5 ACC", "Perm5 BWT", "Perm10 ACC", "Perm10 BWT", "Split ACC", "Split BWT"]].values
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.15, 1.45)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(weight="bold", color="black")
        tbl[(0, j)].set_facecolor("#e6e6e6")
    ax.set_title("Master Continual Learning Summary Across All Benchmarks (5 Seeds: 100-104)", fontsize=12, pad=15, weight="bold")
    plt.tight_layout()
    plt.savefig("outputs/phase5/master_summary.png", dpi=250, bbox_inches="tight")
    plt.close()
    print("Saved outputs/phase5/master_summary.png")

    # 5. Copy over key figures
    figure_copies = [
        ("outputs/phase1/forgetting_curve_perm5.png", "outputs/phase5/forgetting_curve_perm5.png"),
        ("outputs/phase1/heatmaps_perm5.png", "outputs/phase5/heatmaps_perm5.png"),
        ("outputs/phase2/forgetting_curve_perm10.png", "outputs/phase5/forgetting_curve_perm10.png"),
        ("outputs/phase2/heatmaps_perm10.png", "outputs/phase5/heatmaps_perm10.png"),
        ("outputs/phase2/forgetting_curve_split.png", "outputs/phase5/forgetting_curve_split.png"),
        ("outputs/phase2/heatmaps_split.png", "outputs/phase5/heatmaps_split.png"),
        ("outputs/phase3/weight_norms.png", "outputs/phase5/diagnostics_weight_norms.png"),
        ("outputs/phase3/drift_along_d.png", "outputs/phase5/diagnostics_drift_along_d.png"),
        ("outputs/phase3/gated_fraction.png", "outputs/phase5/diagnostics_gated_fraction.png"),
        ("outputs/phase3/layer_heatmap_best_gate.png", "outputs/phase5/layer_heatmap_best_gate.png"),
    ]
    for src, dst in figure_copies:
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")

    print("Results pack built successfully!")

if __name__ == "__main__":
    main()
