# AEGIS: Continual Learning & Catastrophic Forgetting Benchmark

An empirical benchmark evaluating continual learning methods against catastrophic forgetting across sequential Permuted-MNIST tasks using a Multi-Layer Perceptron (MLP) architecture and SGD optimization.

## Benchmark Overview

The benchmark evaluates six continual learning paradigms across **5 sequentially learned permuted tasks** (Permuted-MNIST):

1. **Plain Fine-Tuning (`finetune`)**: Standard unconstrained SGD training on each task sequentially.
2. **Elastic Weight Consolidation (`ewc`)**: Diagonal Fisher information regularizer penalizing changes to parameters critical for previous tasks.
3. **Experience Replay (`replay`)**: Stores an episodic memory buffer of 200 raw exemplars per task, interleaved during training.
4. **Low-Rank Adaptation (`lora`)**: Freezes the base MLP after Task 1 and trains task-specific low-rank matrix pairs ($r=8$) with test-time task-ID conditioning.
5. **Gradient-Conflict Gate (`gate:conflict`)**: Custom directional gate that tracks cumulative parameter displacement $\mathbf{d}$ across past tasks and damps gradient components that oppose prior movement ($\mathbf{g} \cdot \mathbf{d} > 0$).
6. **Gate Magnitude Ablation (`gate:magnitude`)**: Ablation baseline that scales down gradients purely based on historical displacement magnitude, omitting the directional conflict check.
7. **Gate Layer Cosine Variant (`gate:layer_cos`)**: Layer-wise cosine similarity gating.

---

## Evaluation Metrics

- **ACC**: Average accuracy over all 5 tasks at sequence completion.
- **BWT (Backward Transfer)**: Average change in accuracy on earlier tasks after learning subsequent tasks (negative values quantify catastrophic forgetting).
- **FWT (Forward Transfer)**: Performance on subsequent tasks prior to training, relative to an untrained random initialization.
- **NEW**: Average initial accuracy on each task immediately after training on it.

---

## Summary of Results

All values represent **mean ± std (%)** across 3 random seeds:

| Tag | Method | ACC (%) | BWT (%) | FWT (%) | NEW (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **main** | finetune | 90.8 ± 2.4 | -7.7 ± 2.9 | 0.5 ± 3.0 | 97.0 ± 0.0 |
| **main** | ewc | 9.8 ± 0.0 | -21.5 ± 0.1 | -0.7 ± 2.0 | 27.0 ± 0.0 |
| **main** | **replay** | **95.2 ± 0.2** | -2.2 ± 0.3 | 0.7 ± 2.7 | 96.9 ± 0.1 |
| **main** | **lora** | 93.8 ± 0.2 | **0.0 ± 0.0** | 0.9 ± 1.0 | 93.8 ± 0.2 |
| **main** | gate:conflict:1.0 | 13.7 ± 3.0 | -56.9 ± 2.4 | 0.2 ± 2.2 | 59.2 ± 4.2 |
| **main** | gate:magnitude:1.0 | 92.3 ± 0.6 | -3.7 ± 0.6 | 1.0 ± 3.0 | 95.2 ± 0.1 |
| **gate_sweep** | gate:conflict:0.5 | 46.1 ± 2.1 | -57.6 ± 3.0 | -0.6 ± 0.5 | 92.1 ± 0.3 |
| **gate_sweep** | gate:layer_cos:1.0 | 91.1 ± 2.3 | -7.3 ± 2.8 | 0.5 ± 2.9 | 96.9 ± 0.0 |
| **gate_sweep** | gate:magnitude:0.5 | 91.1 ± 2.0 | -6.9 ± 2.4 | 0.6 ± 2.9 | 96.6 ± 0.1 |
| **epochs5** | finetune | 89.8 ± 2.1 | -9.6 ± 2.6 | 0.3 ± 2.9 | **97.5 ± 0.0** |

### Key Findings
- **Replay achieves highest overall accuracy (95.2 ± 0.2% ACC)** by continually anchoring shared representations with raw exemplar buffers.
- **LoRA eliminates forgetting completely (0.0 ± 0.0% BWT)** by isolating task parameters via test-time task conditioning.
- **Magnitude ablation strongly outperforms directional conflict gating**: `gate:magnitude:1.0` achieves 92.3 ± 0.6% ACC vs. 13.7 ± 3.0% for `gate:conflict:1.0`. The elementwise directional conflict check over-restricts gradient updates, severely throttling new-task plasticity.
- **EWC diverged to chance level (9.8 ± 0.0% ACC)** under unnormalized Fisher penalties with standard SGD step size 0.05.

---

## Repository Structure

```
AEGIS/
├── cl.py                   # Core continual learning benchmark & model implementations
├── analyze.py              # Script generating metric heatmaps, bar charts, and summary tables
├── layer_analysis.py       # Hook analyzing layer-wise gradient suppression dynamics
├── run_sweep.sh            # Hyperparameter sweep orchestration script
├── outputs/                # Generated experiment artifacts
│   ├── results_table.csv   # Consolidated metric results across all tags
│   ├── results_table.png   # Rendered high-resolution results table
│   ├── summary.md          # Benchmark findings summary
│   ├── results_explanation.md # Complete poster & examiner documentation
│   ├── heatmaps_*.png      # Accuracy matrix R heatmaps per tag
│   ├── barchart_*.png      # Grouped ACC/BWT/NEW bar charts
│   ├── forgetting_curve_*.png # Task 1 accuracy curves over sequence
│   ├── layer_protection.png# Fraction of gradient norm removed per layer
│   └── results_*.json      # Raw JSON records per run
└── logs/                   # Execution stdout/stderr logs
```

---

## Getting Started

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy pandas
```

### 2. Run Main Experiment
```bash
python cl.py --seeds 3 --epochs 3
```

### 3. Run Hyperparameter Tuning Sweep
```bash
bash run_sweep.sh
```

### 4. Analyze Results
```bash
python analyze.py
python layer_analysis.py
```
