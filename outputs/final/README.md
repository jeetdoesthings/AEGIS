# AEGIS — Displacement-Based Update Gating for Continual Learning

This repository contains a semester-long empirical study of catastrophic forgetting: what happens when a neural network trained on one task is trained on more tasks in sequence, and whether a cheap, buffer-free "gate" on weight updates can reduce the damage.

The short version of what we found:

1. The original idea — a gradient-conflict gate that blocks updates undoing past weight movement — **failed**: it is worse than plain fine-tuning on 5-task Permuted-MNIST and collapses on 10 tasks (41.92% vs 78.21% ACC). Step-level telemetry supports an "asymmetric ratchet" explanation: it blocks steps back toward old positions but lets outward steps run free, so weight norms double and drift away.
2. Repaired variants (magnitude damping, and drift/hybrid gates that also brake steps away from the task-start weights) beat fine-tuning on 10-task Permuted-MNIST by about 5 points of ACC, with consistent per-seed differences — a modest but real gain without storing any old data.
3. The honest headline: **repaired EWC, which also stores no old data, beats every gate variant** on both permuted benchmarks (90.22% vs at most 83.82% on 10 tasks). On Split-MNIST, no method without replay or a test-time task ID gets past 62–64% ACC.

We do not claim to solve catastrophic forgetting. What the semester produced is a fair testbed, a diagnosed failure, tested repairs, and a clear-eyed comparison against stronger baselines.

---

## Results

All runs use 5 held-out test seeds (100–104) on the 10,000-image MNIST test split, after tuning on a separate 5,000-image validation split only. Values are mean ± std (%) over seeds. Source data: [`outputs/phase5/master_summary.csv`](outputs/phase5/master_summary.csv).

| Method | Perm5 ACC | Perm5 BWT | Perm10 ACC | Perm10 BWT | Split ACC | Split BWT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fine-tuning (plain SGD) | 88.97 ± 1.85 | −9.62 ± 2.34 | 78.21 ± 1.68 | −20.80 ± 1.84 | 62.07 ± 1.00 | −46.36 ± 1.23 |
| EWC (repaired, λ=20/100) | 93.76 ± 0.36 | −2.34 ± 0.55 | 90.22 ± 0.87 | −4.17 ± 0.93 | 63.96 ± 1.39 | −43.76 ± 1.72 |
| Replay (buffer=1000) | **95.85 ± 0.11** | −1.18 ± 0.14 | **95.22 ± 0.09** | −1.92 ± 0.14 | 96.85 ± 0.30 | −2.69 ± 0.29 |
| LoRA (r=16/8, oracle ID)\* | 94.30 ± 0.12 | **0.00 ± 0.00** | 94.05 ± 0.05 | **0.00 ± 0.00** | **98.73 ± 0.06** | **0.00 ± 0.00** |
| Gate: conflict (original, β=0.25) | 81.73 ± 1.41 | −17.60 ± 1.84 | 41.92 ± 1.71 | −58.83 ± 1.95 | 62.28 ± 0.98 | −45.88 ± 1.24 |
| Gate: magnitude (ablation, β=1.0) | 91.43 ± 0.86 | −4.39 ± 1.01 | 83.13 ± 1.22 | −12.82 ± 1.33 | 61.90 ± 0.36 | −45.83 ± 0.51 |
| Gate: layer cosine (β=0.5/1.0) | 89.00 ± 2.32 | −9.62 ± 2.98 | 78.80 ± 0.97 | −20.15 ± 1.15 | 62.05 ± 1.02 | −46.39 ± 1.26 |
| Gate: drift (repaired, β=0.1) | 91.89 ± 1.03 | −5.82 ± 1.39 | 83.21 ± 1.26 | −14.98 ± 1.44 | 62.62 ± 1.03 | −45.61 ± 1.27 |
| Gate: hybrid (repaired, β=0.1, γ=0.5) | 90.98 ± 0.90 | −6.64 ± 1.20 | 83.82 ± 0.54 | −13.95 ± 0.64 | 62.78 ± 0.56 | −45.11 ± 0.71 |
| Gate: soft conflict (β=0.05) | 90.68 ± 0.58 | −7.44 ± 0.79 | 80.05 ± 1.66 | −18.58 ± 1.86 | 62.34 ± 1.03 | −45.99 ± 1.28 |

\* LoRA picks the task adapter using the true task ID at test time — a luxury real continual learning does not have, so its row is an upper bound, not a competitor.

Two results are called a tie when their mean ± std intervals overlap; "beats" above means the intervals are separate and the per-seed paired differences agree in sign across all five seeds.

---

## How the experiments are kept fair

- MNIST's 60,000 training images are split 55,000 train / 5,000 validation (fixed seed 42). The 10,000 test images are only touched by the final runs.
- Every method gets the same tuning budget (at most 6 configurations, seeds 0–2) on validation. The winners are locked into `configs/locked_*.json` before any test run.
- Training setup is identical everywhere: MLP 784-256-256-C from scratch, plain SGD, lr 0.05, batch 128, 3 epochs per task.
- A divergence guard halts any run whose loss goes non-finite (this is how the original EWC failure was caught — see [`outputs/failed/`](outputs/failed/README.md)).

---

## Repository layout

```
AEGIS/
├── cl.py                       # Original prototype, kept frozen as a reference
├── cl2.py                      # Main experiment engine (splits, guards, all methods)
├── tune_val.py                 # Validation sweep for Perm5 (seeds 0-2)
├── run_final_perm5.py          # Final Perm5 evaluation (test seeds 100-104)
├── run_perm10.py               # Final Perm10 evaluation
├── tune_and_run_split.py       # Tune + evaluate Split-MNIST
├── tune_and_run_repaired.py    # Tune + evaluate drift / hybrid / soft conflict
├── diagnostics.py              # Step-level weight norms, drift along d, gated fraction
├── layer_analysis.py           # Per-layer gated-fraction heatmap
├── build_results_pack.py       # Consolidates tables and figures into outputs/phase5/
├── scripts/                    # Tooling for the standalone overview PDFs
├── tests/                      # Unit and regression tests (pytest)
├── configs/                    # Locked best configurations per benchmark
├── logs/                       # Run logs
├── outputs/
│   ├── phase1/                 # EWC diagnosis, validation sweep, Perm5 final runs
│   ├── phase2/                 # Perm10 and Split-MNIST final runs
│   ├── phase3/                 # Diagnostics, ratchet analysis, repaired gates
│   ├── phase5/                 # Report, poster text, Q&A, tables, figures
│   ├── final/                  # Frozen review deliverables (reports, poster, figures)
│   ├── audit/                  # Number recomputation, claims ledger, regression runs
│   └── failed/                 # Diverged/invalid runs, kept for the record
├── archive/phase0/             # First prototype experiments (superseded; see note below)
├── PLAN.md                     # Semester plan the work followed
├── STATUS.md                   # Research log, oldest to newest
└── requirements.txt
```

A note on [`archive/phase0/`](archive/phase0/): the very first experiments tuned hyperparameters on the test set — a mistake we found and corrected. Those raw results are kept (not deleted) so the improvement in methodology is visible, but nothing from there is cited as a current result.

---

## Reproducing

### 1. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
MNIST downloads automatically via torchvision into `data/` on first use.

### 2. Smoke test (~15 s, 1 seed, 1 epoch)
```bash
python cl2.py --seeds 1 --epochs 1
```

### 3. Full pipeline (CPU; a few minutes per benchmark)
```bash
python tune_val.py               # validation sweep, Perm5
python run_final_perm5.py        # final Perm5, 5 test seeds
python run_perm10.py             # final Perm10
python tune_and_run_split.py     # Split-MNIST
python tune_and_run_repaired.py  # repaired gate variants
python diagnostics.py            # ratchet telemetry
python build_results_pack.py     # rebuild tables/figures in outputs/phase5/
```

### 4. Tests
```bash
python -m pytest tests/          # 15 tests: gate mechanics, metrics, split isolation
```

---

## Documents

- **Semester report:** [`outputs/final/PROJECT_REPORT.md`](outputs/final/PROJECT_REPORT.md) (PDF alongside)
- **Audit report:** [`outputs/final/AUDIT_REPORT.md`](outputs/final/AUDIT_REPORT.md) (PDF alongside)
- **Full paper draft:** [`outputs/phase5/report.md`](outputs/phase5/report.md)
- **Claim audit:** [`outputs/phase5/claim_audit.md`](outputs/phase5/claim_audit.md)
- **Poster text / 60-second script / Q&A:** [`poster_text.md`](outputs/phase5/poster_text.md) · [`script_60s.md`](outputs/phase5/script_60s.md) · [`qa.md`](outputs/phase5/qa.md)
- **Figure captions:** [`outputs/phase5/captions.md`](outputs/phase5/captions.md)
- **Short overview / long overview PDFs:** [`outputs/final/AEGIS_Simplified_Project_Overview.pdf`](outputs/final/AEGIS_Simplified_Project_Overview.pdf) · [`outputs/final/AEGIS_Master_Project_Report.pdf`](outputs/final/AEGIS_Master_Project_Report.pdf)

## Status and what's next

This semester covered the vision testbed, the baselines, the conflict-gate failure and its diagnosis, and the repaired variants. Next semester: SI/MAS/A-GEM/L2-SP baselines, gate+EWC, a GPT-2 experiment with a gate on Adam's final update, and a written thesis. See [`STATUS.md`](STATUS.md) for the full log and [`outputs/final/PROJECT_REPORT.md`](outputs/final/PROJECT_REPORT.md) §13 for the week-by-week plan.
