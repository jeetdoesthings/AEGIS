import os, sys, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None, help="Specific tag to analyze, or None for all")
    args = parser.parse_args()

    files = sorted(glob.glob("outputs/results_*.json"))
    if not files:
        print("No results_*.json files found in outputs/")
        return

    records = []
    tag_data = {}

    for f in files:
        tag = os.path.basename(f).replace("results_", "").replace(".json", "")
        if args.tag and tag != args.tag:
            continue
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        tag_data[tag] = data

        for m, rs in data.items():
            accs = [r["ACC"] * 100 for r in rs]
            bwts = [r["BWT"] * 100 for r in rs]
            fwts = [r["FWT"] * 100 for r in rs]
            news = [r["NEW"] * 100 for r in rs]
            records.append({
                "tag": tag,
                "method": m,
                "ACC_mean": float(np.mean(accs)), "ACC_std": float(np.std(accs)),
                "BWT_mean": float(np.mean(bwts)), "BWT_std": float(np.std(bwts)),
                "FWT_mean": float(np.mean(fwts)), "FWT_std": float(np.std(fwts)),
                "NEW_mean": float(np.mean(news)), "NEW_std": float(np.std(news)),
            })

    if not records:
        print("No matching data found.")
        return

    df = pd.DataFrame(records)

    # Format text table with bolding best ACC and best BWT
    best_acc = df["ACC_mean"].max()
    best_bwt = df["BWT_mean"].max()

    print(f"\n{'tag':<15}{'method':<22}{'ACC':>16}{'BWT':>16}{'FWT':>16}{'NEW':>16}")
    print("-" * 101)
    for _, row in df.iterrows():
        acc_s = f"{row['ACC_mean']:6.1f}+-{row['ACC_std']:4.1f}"
        if abs(row['ACC_mean'] - best_acc) < 1e-5:
            acc_s = f"*{acc_s}*"
        bwt_s = f"{row['BWT_mean']:6.1f}+-{row['BWT_std']:4.1f}"
        if abs(row['BWT_mean'] - best_bwt) < 1e-5:
            bwt_s = f"*{bwt_s}*"
        fwt_s = f"{row['FWT_mean']:6.1f}+-{row['FWT_std']:4.1f}"
        new_s = f"{row['NEW_mean']:6.1f}+-{row['NEW_std']:4.1f}"
        print(f"{row['tag']:<15}{row['method']:<22}{acc_s:>16}{bwt_s:>16}{fwt_s:>16}{new_s:>16}")

    # Save CSV
    csv_df = pd.DataFrame({
        "tag": df["tag"],
        "method": df["method"],
        "ACC": [f"{m:.2f} +- {s:.2f}" for m, s in zip(df["ACC_mean"], df["ACC_std"])],
        "BWT": [f"{m:.2f} +- {s:.2f}" for m, s in zip(df["BWT_mean"], df["BWT_std"])],
        "FWT": [f"{m:.2f} +- {s:.2f}" for m, s in zip(df["FWT_mean"], df["FWT_std"])],
        "NEW": [f"{m:.2f} +- {s:.2f}" for m, s in zip(df["NEW_mean"], df["NEW_std"])],
    })
    csv_path = "outputs/results_table.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV table to {csv_path}")

    # Generate results table image
    fig, ax = plt.subplots(figsize=(10, max(2, len(df) * 0.45 + 1.2)))
    ax.axis("off")
    table_data = []
    for _, row in df.iterrows():
        acc_str = f"{row['ACC_mean']:.1f} ± {row['ACC_std']:.1f}"
        bwt_str = f"{row['BWT_mean']:.1f} ± {row['BWT_std']:.1f}"
        fwt_str = f"{row['FWT_mean']:.1f} ± {row['FWT_std']:.1f}"
        new_str = f"{row['NEW_mean']:.1f} ± {row['NEW_std']:.1f}"
        table_data.append([row['tag'], row['method'], acc_str, bwt_str, fwt_str, new_str])

    col_labels = ["Tag", "Method", "ACC (%)", "BWT (%)", "FWT (%)", "NEW (%)"]
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.4)
    # Highlight best
    for i, row in df.iterrows():
        if abs(row['ACC_mean'] - best_acc) < 1e-5:
            tbl[(i + 1, 2)].set_text_props(weight="bold", color="darkblue")
        if abs(row['BWT_mean'] - best_bwt) < 1e-5:
            tbl[(i + 1, 3)].set_text_props(weight="bold", color="darkgreen")
    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(weight="bold")
    plt.tight_layout()
    plt.savefig("outputs/results_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved outputs/results_table.png")

    # Generate heatmaps and bar charts for available tags
    for tag, data in tag_data.items():
        methods = list(data.keys())
        n_m = len(methods)
        if n_m == 0:
            continue

        # a) Heatmaps of accuracy matrix R averaged over seeds
        cols = min(3, n_m)
        rows = int(np.ceil(n_m / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8), squeeze=False)
        for idx, m in enumerate(methods):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]
            R_mean = np.mean([np.array(run["R"]) for run in data[m]], axis=0) * 100
            im = ax.imshow(R_mean, cmap="YlGnBu", vmin=0, vmax=100)
            ax.set_title(m, fontsize=11, fontweight="bold")
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

        # Hide unused subplots
        for idx in range(n_m, rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        fig.suptitle(f"Accuracy Matrix R (%) — Tag: {tag}", fontsize=14, y=1.01)
        fig.tight_layout()
        heatmap_path = f"outputs/heatmaps_{tag}.png"
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {heatmap_path}")

        # b) Grouped bar chart of ACC, BWT, NEW per method with std error bars
        sub_df = df[df["tag"] == tag]
        fig, ax = plt.subplots(figsize=(max(7, len(sub_df) * 1.5), 5))
        x = np.arange(len(sub_df))
        width = 0.25

        ax.bar(x - width, sub_df["ACC_mean"], width, yerr=sub_df["ACC_std"], label="ACC", capsize=4, color="#2b5c8f")
        ax.bar(x, sub_df["BWT_mean"], width, yerr=sub_df["BWT_std"], label="BWT", capsize=4, color="#d95f02")
        ax.bar(x + width, sub_df["NEW_mean"], width, yerr=sub_df["NEW_std"], label="NEW", capsize=4, color="#2ca02c")

        ax.set_ylabel("Percent (%)", fontsize=11)
        ax.set_title(f"Method Comparison (ACC, BWT, NEW) — Tag: {tag}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub_df["method"], rotation=25, ha="right", fontsize=10)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        barchart_path = f"outputs/barchart_{tag}.png"
        fig.savefig(barchart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {barchart_path}")

if __name__ == "__main__":
    main()
