# AEGIS: Continual Learning with Directional & Magnitude Gradient Gating

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c.svg)](https://pytorch.org/)

An academically rigorous empirical investigation into catastrophic forgetting, directional gradient conflict gating, and parameter-drift dynamics across sequential neural network tasks.

---

## 📌 Executive Summary

Continual learning in neural networks is fundamentally constrained by catastrophic forgetting—the overwriting of past representations when training on new data distributions. This project investigates whether parameter-space gradient gating can protect historical knowledge without storing past exemplars (replay buffers).

Through exhaustive, fair-budget evaluations on held-out test splits across 5 independent random seeds (100–104), this project establishes:
1. **The Directional Ratchet Failure:** The original directional gradient conflict gate (`gate:conflict`) fails catastrophically, collapsing to **$41.92 \pm 1.71\%$ accuracy** on 10 Permuted-MNIST tasks ($-36.29 \pm 2.82\%$ below plain fine-tuning). Step-level parameter diagnostics support the *asymmetric ratchet mechanism*: selectively zeroing updates opposing historical displacement ($g \cdot d > 0$) prevents restoring steps while letting outward steps proceed unhindered, causing parameter norms to double ($l1.weight$ L2 norm 22.70 vs 11.79 for fine-tuning) and run away along $d$.
2. **Magnitude Outperforms Direction:** Ignoring update direction and dampening weights based purely on historical displacement magnitude (`gate:magnitude`) prevents runaway inflation and outperforms fine-tuning on 10 tasks ($83.13 \pm 1.22\%$ vs $78.21 \pm 1.68\%$).
3. **Anti-Ratchet Repairs:** Formulating an anti-drift gradient gate (`gate:drift`, which damps updates that increase distance from starting weights $e = \theta - \theta_{start}$) and a hybrid gate (`gate:hybrid`) eliminates the ratchet runaway. On 10 tasks, `gate:hybrid` ($83.82 \pm 0.54\%$) and `gate:drift` ($83.21 \pm 1.26\%$) **statistically outperform plain fine-tuning ($78.21 \pm 1.68\%$) by $>+5\%$ ACC** without replay buffers.
4. **The Output Head Bottleneck:** `[MEASURED]` On domain-incremental Split-MNIST, all buffer-free methods tie around $62\%$ ACC. `[HYPOTHESIS]` This collapse is attributed to representational collision at the shared 2-way output head, whereas exemplar replay ($96.85\%$) and parameter-isolated LoRA ($98.73\%$) retain near-optimal performance.

> **Note on Phase 4 (Language Model Experiment):** Per the go/no-go gate rule in `AGENT_PLAN.md`, Phase 4 was skipped during local execution on CPU hardware and is documented strictly as Future Work. Training a 124M-parameter GPT-2 model across 4 domains on CPU without GPU acceleration is computationally intractable. Phases 0, 1, 2, 3, and 5 are fully completed.

---

## 📊 Master Benchmark Results

*All evaluations conducted on held-out test splits across 5 random seeds (100, 101, 102, 103, 104). Values reported as mean ± std (%). Source: [`outputs/phase5/master_summary.csv`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/master_summary.csv).*

| Method | Perm5 ACC (%) | Perm5 BWT (%) | Perm10 ACC (%) | Perm10 BWT (%) | Split-MNIST ACC (%) | Split-MNIST BWT (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuning (Plain SGD)** | $88.97 \pm 1.85$ | $-9.62 \pm 2.34$ | $78.21 \pm 1.68$ | $-20.80 \pm 1.84$ | $62.07 \pm 1.00$ | $-46.36 \pm 1.23$ |
| **EWC (repaired, $\lambda=20/100$)** | $93.76 \pm 0.36$ | $-2.34 \pm 0.55$ | $90.22 \pm 0.87$ | $-4.17 \pm 0.93$ | $63.96 \pm 1.39$ | $-43.76 \pm 1.72$ |
| **Replay (buffer=1000)** | $\mathbf{95.85 \pm 0.11}$ | $-1.18 \pm 0.14$ | $\mathbf{95.22 \pm 0.09}$ | $-1.92 \pm 0.14$ | $96.85 \pm 0.30$ | $-2.69 \pm 0.29$ |
| **LoRA ($r=16/8$, oracle ID)\*** | $94.30 \pm 0.12$ | $\mathbf{0.00 \pm 0.00}$ | $94.05 \pm 0.05$ | $\mathbf{0.00 \pm 0.00}$ | $\mathbf{98.73 \pm 0.06}$ | $\mathbf{0.00 \pm 0.00}$ |
| **Gate: Conflict (original, $\beta=0.25$)** | $81.73 \pm 1.41$ | $-17.60 \pm 1.84$ | $41.92 \pm 1.71$ | $-58.83 \pm 1.95$ | $62.28 \pm 0.98$ | $-45.88 \pm 1.24$ |
| **Gate: Magnitude (ablation, $\beta=1.0$)** | $91.43 \pm 0.86$ | $-4.39 \pm 1.01$ | $83.13 \pm 1.22$ | $-12.82 \pm 1.33$ | $61.90 \pm 0.36$ | $-45.83 \pm 0.51$ |
| **Gate: Layer Cosine ($\beta=0.5/1.0$)** | $89.00 \pm 2.32$ | $-9.62 \pm 2.98$ | $78.80 \pm 0.97$ | $-20.15 \pm 1.15$ | $62.05 \pm 1.02$ | $-46.39 \pm 1.26$ |
| **Gate: Drift (repaired, $c=0.1$)** | $91.89 \pm 1.03$ | $-5.82 \pm 1.39$ | $83.21 \pm 1.26$ | $-14.98 \pm 1.44$ | $62.62 \pm 1.03$ | $-45.61 \pm 1.27$ |
| **Gate: Hybrid (repaired, $c=0.1$)** | $90.98 \pm 0.90$ | $-6.64 \pm 1.20$ | $83.82 \pm 0.54$ | $-13.95 \pm 0.64$ | $62.78 \pm 0.56$ | $-45.11 \pm 0.71$ |
| **Gate: Soft Conflict ($\beta=0.1$)** | $90.68 \pm 0.58$ | $-7.44 \pm 0.79$ | $80.05 \pm 1.66$ | $-18.58 \pm 1.86$ | $62.34 \pm 1.03$ | $-45.99 \pm 1.28$ |

*\*LoRA utilizes an oracle task ID at test time to select task-specific adapters.*

---

## 🛠️ Repository Structure

```
AEGIS/
├── cl.py                       # Frozen original reference code
├── cl2.py                      # Enhanced engine (splits, guards, diagnostics, repaired gates)
├── tune_val.py                 # Hyperparameter sweep across 25 configs on validation split
├── run_final_perm5.py          # Final 5-seed evaluation on 5-task Permuted-MNIST
├── run_perm10.py               # Final 5-seed evaluation on 10-task Permuted-MNIST
├── tune_and_run_split.py       # Validation tuning and test evaluation on Split-MNIST
├── diagnostics.py              # Step-level parameter telemetry & ratchet hypothesis test
├── tune_and_run_repaired.py    # Tuning & 5-seed evaluation of repaired gate variants
├── build_results_pack.py       # Consolidates all tables, plots, and metadata for deliverables
├── configs/                    # Locked JSON hyperparameters per benchmark
│   ├── locked_perm5.json       # Best configs for 5-task Permuted-MNIST
│   ├── locked_split.json       # Best configs for Split-MNIST
│   └── locked_repaired.json    # Best configs for repaired gates
├── outputs/
│   ├── phase1/                 # EWC diagnosis, validation sweep, 5-task Permuted-MNIST test
│   ├── phase2/                 # 10-task Permuted-MNIST and Split-MNIST test results
│   ├── phase3/                 # Diagnostic CSVs, ratchet hypothesis, findings, layer heatmap
│   └── phase5/                 # Final deliverables pack:
│       ├── report.md           # Full academic research paper draft
│       ├── claim_audit.md      # Statistical tie audit log verifying all assertions
│       ├── poster_text.md      # Publication-ready poster text
│       ├── qa.md               # 15 examiner defense questions with exact numbers
│       ├── script_60s.md       # 60-second spoken poster script
│       ├── captions.md         # Detailed figure and table captions
│       ├── master_summary.*    # Consolidated 10-method comparison (CSV + PNG)
│       ├── table_*.*           # Per-benchmark publication tables (CSV + PNG)
│       └── *.png               # Forgetting curves, heatmaps, diagnostics, and layer plots
└── STATUS.md                   # Real-time task progress log appended after every milestone
```

---

## 🚀 Reproduction Commands

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy pandas
```

### 2. Fast Smoke Test (1 seed, 1 epoch)
```bash
python cl2.py --seeds 1 --epochs 1
```

### 3. Permuted-MNIST 5-Task Pipeline
```bash
# Tune all 7 methods across 25 configs on validation split (seeds 0-2)
python tune_val.py

# Evaluate locked configs across 5 fresh test seeds (100-104)
python run_final_perm5.py
```

### 4. Permuted-MNIST 10-Task Evaluation
```bash
# Evaluate locked configs on 10 sequential tasks across 5 test seeds (100-104)
python run_perm10.py
```

### 5. Split-MNIST Evaluation
```bash
# Tune on validation and evaluate 5 test seeds on domain-incremental digit pairs
python tune_and_run_split.py
```

### 6. Mechanistic Diagnostics & Hypothesis Test
```bash
# Log step-by-step weight norms, drift along d, and test ratchet hypothesis
python diagnostics.py
```

### 7. Repaired Gates Evaluation
```bash
# Tune repaired gates (drift, hybrid, soft) and evaluate across 5 seeds on all benchmarks
python tune_and_run_repaired.py
```

### 8. Build Final Phase 5 Deliverables Pack
```bash
# Regenerate master tables, plots, and figures in outputs/phase5/
python build_results_pack.py
```

---

## 📚 Deliverables & Documentation

- **Full Academic Report:** [`outputs/phase5/report.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md)
- **Claim Audit & Statistical Verification:** [`outputs/phase5/claim_audit.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/claim_audit.md)
- **Poster Presentation Text:** [`outputs/phase5/poster_text.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/poster_text.md)
- **15 Examiner Defense Q&A:** [`outputs/phase5/qa.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/qa.md)
- **60-Second Spoken Poster Script:** [`outputs/phase5/script_60s.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/script_60s.md)
- **Figure & Table Captions:** [`outputs/phase5/captions.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/captions.md)
- **Milestone Log:** [`STATUS.md`](file:///Users/jeet/Desktop/AEGIS/STATUS.md)
