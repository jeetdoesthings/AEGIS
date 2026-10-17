import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cl

class LoggingGate(cl.Gate):
    last_instance = None

    def __init__(self, model, mode, beta, decay):
        super().__init__(model, mode, beta, decay)
        self.task_idx = 0
        self.param_names = []
        self.history = {}  # (task_idx, name) -> list of fractions removed
        LoggingGate.last_instance = self

    def begin_task(self, model):
        super().begin_task(model)
        self.param_names = [name for name, _ in model.named_parameters()]
        self.task_idx += 1

    def apply(self, model):
        norms_before = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                norms_before[name] = p.grad.norm().item()

        super().apply(model)

        for name, p in model.named_parameters():
            if p.grad is not None and name in norms_before:
                nb = norms_before[name]
                if nb > 1e-12:
                    na = p.grad.norm().item()
                    frac = 1.0 - na / nb
                else:
                    frac = 0.0
                key = (self.task_idx, name)
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(frac)

def main():
    cl.Gate = LoggingGate

    args = argparse.Namespace(
        tasks=5,
        epochs=3,
        lr=0.05,
        bs=128,
        seeds=3,
        ewc_lambda=5000,
        buf=200,
        beta=1.0,
        decay=1.0,
    )

    print("Loading MNIST data...")
    data = cl.load_mnist()

    print("Running gate:conflict:1.0 with LoggingGate...")
    res = cl.run("gate:conflict:1.0", args, 0, data)
    print("Run completed. ACC:", res["ACC"])

    gate = LoggingGate.last_instance
    if gate is None or not gate.history:
        print("Error: No logging history recorded.")
        return

    row_names = ["l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]
    col_tasks = [2, 3, 4, 5]

    heatmap_data = np.zeros((len(row_names), len(col_tasks)))

    for i, r_name in enumerate(row_names):
        for j, t_idx in enumerate(col_tasks):
            key = (t_idx, r_name)
            if key in gate.history and len(gate.history[key]) > 0:
                heatmap_data[i, j] = np.mean(gate.history[key]) * 100
            else:
                heatmap_data[i, j] = 0.0

    print("Heatmap Data (% gradient removed):")
    for i, r_name in enumerate(row_names):
        vals = "  ".join(f"{heatmap_data[i, j]:5.1f}%" for j in range(len(col_tasks)))
        print(f"{r_name:12s} | {vals}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap_data, cmap="Blues", vmin=0, vmax=max(50, heatmap_data.max() * 1.1))

    ax.set_xticks(range(len(col_tasks)))
    ax.set_yticks(range(len(row_names)))
    ax.set_xticklabels([f"Task {t}" for t in col_tasks], fontsize=10)
    ax.set_yticklabels(row_names, fontsize=10)
    ax.set_xlabel("Task Index", fontsize=11)
    ax.set_ylabel("Layer / Parameter", fontsize=11)
    ax.set_title("Fraction of Gradient Removed by Gate (%)", fontsize=12, fontweight="bold")

    for i in range(len(row_names)):
        for j in range(len(col_tasks)):
            val = heatmap_data[i, j]
            color = "white" if val > 25 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = "outputs/layer_protection.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
