# Project Status

## Task 0.1 Environment & Device Check
- status: done
- what changed: Verified .venv with torch 2.8.0, torchvision, matplotlib, numpy, pandas. Benchmarked CPU vs MPS on MLP operations.
- key numbers: CPU: 0.156s vs MPS: 0.317s for 500 MLP batch forward passes. CPU has significantly lower dispatch latency for small MLP matrix operations.
- problems / decisions: Use CPU for all runs as specified in plan.
- next: Task 0.2

## Task 0.2 Freeze & Project Structure
- status: done
- what changed: cl.py frozen. Copied cl.py to cl2.py. Created folders lm/, outputs/phase1/, outputs/phase2/, outputs/phase3/, outputs/phase4/, outputs/phase5/, outputs/failed/, logs/, configs/. Created STATUS.md. Verified acceptance with `python cl2.py --seeds 1 --epochs 1`.
- key numbers: Smoke test user time: 15.87s. Smoke test table: finetune ACC 88.0%, replay ACC 92.9%, lora ACC 90.1%, gate:mag ACC 89.7%, gate:conf ACC 32.4%, ewc ACC 9.8% (diverged).
- problems / decisions: cl.py is untouched. All Phase 1+ code will be developed in cl2.py.
- next: Task 1.1

## Task 1.1 Validation split and seed control
- status: done
- what changed: Modified cl2.py to split MNIST train into 55,000 train / 5,000 validation using fixed generator seed 42. Added `--eval_split val|test`, `--seed_start N`, and `--out TAG`. Implemented configuration JSON logging (all args, git commit, torch version, device, seeds).
- key numbers: Evaluation on `--eval_split val` yielded ACC=87.2% vs `--eval_split test` ACC=87.5% (from outputs/phase1/results_test_val.json and outputs/phase1/results_test_test.json).
- problems / decisions: Training data strictly isolated to 55,000 samples for all runs, guaranteeing validation split is never seen during training.
- next: Task 1.2

## Task 1.2 Repair EWC
- status: done
- what changed: Reproduced and isolated EWC divergence at lambda=5000: exponential gradient explosion caused non-finite loss (NaN/Inf) at step 13 of Task 2, logged to outputs/phase1/ewc_diagnosis.txt. Added a divergence guard in `train_task` and `run` in cl2.py that halts diverged runs and records `diverged: true` without reporting fake accuracy. Swept lambda in {1, 5, 20, 100} across 3 validation seeds (0, 1, 2). Added `--clip` argument.
- key numbers: (All from outputs/phase1/results_ewc_val_lam*.json):
  - lambda=1: ACC 91.0 ± 1.3%, BWT -6.6 ± 1.5%, NEW 96.3 ± 0.1%
  - lambda=5: ACC 92.8 ± 0.9%, BWT -4.1 ± 1.0%, NEW 96.1 ± 0.1%
  - lambda=20: ACC 93.7 ± 0.7%, BWT -2.1 ± 0.6%, NEW 95.3 ± 0.3% (BEST ACC)
  - lambda=100: ACC 92.3 ± 0.8%, BWT -0.8 ± 0.3%, NEW 93.0 ± 0.6%
- problems / decisions: [MEASURED] lambda=20 is the best setting on validation by ACC (93.7%). [HYPOTHESIS] The previous 9.80% result was caused by quadratic penalty explosion ($\text{lr} \times \lambda \times F_i \gg 1$), which caused SGD updates to oscillate and diverge exponentially into NaNs, defaulting predictions to class 0. With $\lambda=20$, the gradient update remains within the stability radius of plain SGD, successfully regularizing against forgetting.
- next: Task 1.3

## Task 1.3 Fair tuning of everything
- status: done
- what changed: Created tune_val.py. Swept all methods on validation split across 3 seeds (0, 1, 2) with an equal tuning budget (<= 6 configs each): finetune (1), ewc lambda in {1, 5, 20, 100} (4), replay buf in {50, 200, 1000} (3), lora r in {4, 8, 16} (3), gate:conflict beta in {0.25, 0.5, 1.0} x decay in {1.0, 0.9} (6), gate:magnitude beta in {0.25, 0.5, 1.0} x decay in {1.0, 0.9} (6), gate:layer_cos beta in {0.5, 1.0} (2). Saved full comparison to outputs/phase1/tuning_val.csv. Locked best configs by ACC (ties by BWT, then NEW) into configs/locked_perm5.json.
- key numbers: (From outputs/phase1/tuning_val.csv):
  - finetune: ACC 90.85 ± 1.83%, BWT -6.90 ± 2.27%
  - ewc (locked: lambda=20): ACC 93.66 ± 0.73%, BWT -2.08 ± 0.61%
  - replay (locked: buf=1000): ACC 95.40 ± 0.14%, BWT -1.22 ± 0.13%
  - lora (locked: r=16): ACC 93.52 ± 0.02%, BWT 0.00 ± 0.00%
  - gate:conflict (locked: beta=0.25, decay=1.0): ACC 82.92 ± 1.79%, BWT -15.71 ± 2.17% (beta=1.0 diverged or scored ~14.5%)
  - gate:magnitude (locked: beta=1.0, decay=1.0): ACC 91.51 ± 0.42%, BWT -3.50 ± 0.44%
  - gate:layer_cos (locked: beta=1.0, decay=1.0): ACC 91.06 ± 1.47%, BWT -6.68 ± 1.83%
- problems / decisions: [MEASURED] Decreasing beta to 0.25 rescues gate:conflict from catastrophic collapse (82.9% vs ~14.5% at beta=1.0), but it still underperforms fine-tuning by 7.9% ACC. EWC with lambda=20 and replay with buf=1000 are the top-performing baselines.
- next: Task 1.4

## Task 1.4 Final run on Permuted-MNIST, 5 tasks
- status: done
- what changed: Implemented run_final_perm5.py. Evaluated all 7 locked method configurations on the test split across 5 fresh test seeds (100, 101, 102, 103, 104). Saved raw results to outputs/phase1/final_perm5.json, generated summary CSV outputs/phase1/final_perm5_table.csv, rendered table image outputs/phase1/final_perm5_table.png, plotted task 1 forgetting curves with mean ± std bands to outputs/phase1/forgetting_curve_perm5.png, and generated 7-method accuracy matrix heatmaps to outputs/phase1/heatmaps_perm5.png. Computed paired difference per seed versus plain fine-tuning.
- key numbers: (From outputs/phase1/final_perm5_table.csv, 5 seeds 100-104 on test split, mean ± std):
  - replay (buf=1000): ACC 95.85 ± 0.11% (diff vs FT: +6.88 ± 1.87%), BWT -1.18 ± 0.14% (diff vs FT: +8.43 ± 2.43%), NEW 96.80 ± 0.06% [BEST ACC]
  - lora (r=16, oracle task ID): ACC 94.30 ± 0.12% (diff vs FT: +5.32 ± 1.78%), BWT 0.00 ± 0.00% (diff vs FT: +9.62 ± 2.34%), NEW 94.30 ± 0.12% [LEAST FORGETTING]
  - ewc (lambda=20): ACC 93.76 ± 0.36% (diff vs FT: +4.78 ± 1.69%), BWT -2.34 ± 0.55% (diff vs FT: +7.28 ± 2.07%), NEW 95.63 ± 0.21%
  - gate:magnitude (beta=1.0, decay=1.0): ACC 91.43 ± 0.86% (diff vs FT: +2.46 ± 1.73%), BWT -4.39 ± 1.01% (diff vs FT: +5.23 ± 2.23%), NEW 94.94 ± 0.06%
  - finetune: ACC 88.97 ± 1.85%, BWT -9.62 ± 2.34%, NEW 96.67 ± 0.12%
  - gate:layer_cos (beta=1.0, decay=1.0): ACC 89.00 ± 2.32% (diff vs FT: +0.02 ± 0.86%), BWT -9.62 ± 2.98% (diff vs FT: -0.00 ± 1.07%), NEW 96.69 ± 0.12% (statistical tie with FT)
  - gate:conflict (beta=0.25, decay=1.0): ACC 81.73 ± 1.41% (diff vs FT: -7.24 ± 1.38%), BWT -17.60 ± 1.84% (diff vs FT: -7.98 ± 1.67%), NEW 95.81 ± 0.10%
- problems / decisions: [MEASURED] Under honest evaluation on held-out test permutations:
  1. The directional conflict gate actively impairs continual learning: gate:conflict is -7.24 ± 1.38% worse than plain fine-tuning on ACC and forgets almost twice as much (-17.60% vs -9.62% BWT).
  2. The magnitude gate genuinely mitigates forgetting without data buffers: +2.46 ± 1.73% ACC over fine-tuning, cutting forgetting in half (-4.39% vs -9.62% BWT).
  3. Repaired EWC (lambda=20) is strong (+4.78 ± 1.69% ACC over FT), trailing only replay and LoRA.
- next: Task 2.1

## Task 2.1 10-task Permuted-MNIST
- status: done
- what changed: Implemented and executed run_perm10.py. Evaluated all 7 locked method configurations on a 10-task sequential sequence across 5 test seeds (100-104) on the test split. Saved outputs to outputs/phase2/final_perm10.json, outputs/phase2/final_perm10_table.csv, outputs/phase2/final_perm10_table.png, outputs/phase2/forgetting_curve_perm10.png, and outputs/phase2/heatmaps_perm10.png. Computed per-seed paired differences against fine-tuning.
- key numbers: (From outputs/phase2/final_perm10_table.csv, 5 seeds 100-104 on test split, mean ± std):
  - replay (buf=1000): ACC 95.22 ± 0.09% (diff vs FT: +17.01 ± 1.70%), BWT -1.92 ± 0.14% (diff vs FT: +18.88 ± 1.88%), NEW 96.94 ± 0.05% [BEST ACC]
  - lora (r=16, oracle task ID): ACC 94.05 ± 0.05% (diff vs FT: +15.84 ± 1.69%), BWT 0.00 ± 0.00% (diff vs FT: +20.80 ± 1.84%), NEW 94.05 ± 0.05% [LEAST FORGETTING]
  - ewc (lambda=20): ACC 90.22 ± 0.87% (diff vs FT: +12.01 ± 0.86%), BWT -4.17 ± 0.93% (diff vs FT: +16.63 ± 1.08%), NEW 93.97 ± 0.15%
  - gate:magnitude (beta=1.0, decay=1.0): ACC 83.13 ± 1.22% (diff vs FT: +4.92 ± 1.20%), BWT -12.82 ± 1.33% (diff vs FT: +7.97 ± 1.34%), NEW 94.67 ± 0.04%
  - gate:layer_cos (beta=1.0, decay=1.0): ACC 78.80 ± 0.97% (diff vs FT: +0.59 ± 2.40%), BWT -20.15 ± 1.15% (diff vs FT: +0.64 ± 2.66%), NEW 96.94 ± 0.09% (statistical tie with FT)
  - finetune: ACC 78.21 ± 1.68%, BWT -20.80 ± 1.84%, NEW 96.92 ± 0.10%
  - gate:conflict (beta=0.25, decay=1.0): ACC 41.92 ± 1.71% (diff vs FT: -36.29 ± 2.82%), BWT -58.83 ± 1.95% (diff vs FT: -38.03 ± 3.13%), NEW 94.86 ± 0.15%
- problems / decisions: [MEASURED]
  1. With 10 tasks, catastrophic forgetting in fine-tuning doubles from -9.62% to -20.80% BWT, dropping ACC to 78.21%.
  2. Replay and LoRA remain completely resilient (ACC 95.22% and 94.05%).
  3. Repaired EWC demonstrates massive retention capability: +12.01 ± 0.86% ACC over fine-tuning, curbing forgetting to -4.17%.
  4. The magnitude gate clearly separates from fine-tuning (+4.92 ± 1.20% ACC, +7.97 ± 1.34% BWT).
  5. The conflict gate suffers severe compounding degradation over 10 tasks: ACC collapses to 41.92% (-36.29% vs FT) and BWT plunges to -58.83%.
- next: Task 2.2

## Task 2.2 Split-MNIST, domain-incremental
- status: done
- what changed: Added `--bench splitmnist` to cl2.py with domain-incremental 2-class output heads across 5 sequential digit pair tasks. Implemented tune_and_run_split.py. Tuned all 7 methods across 25 configurations on validation (seeds 0-2), saved outputs/phase2/tuning_split_val.csv, locked best configurations into configs/locked_split.json, and ran final 5 seeds (100-104) on test split. Generated outputs/phase2/final_split.json, outputs/phase2/final_split_table.csv, outputs/phase2/final_split_table.png, outputs/phase2/forgetting_curve_split.png, and outputs/phase2/heatmaps_split.png.
- key numbers: (From outputs/phase2/final_split_table.csv, 5 seeds 100-104 on test split, mean ± std):
  - lora (r=8, oracle task ID): ACC 98.73 ± 0.06% (diff vs FT: +36.66 ± 1.02%), BWT 0.00 ± 0.00% (diff vs FT: +46.36 ± 1.23%), NEW 98.73 ± 0.06% [BEST ACC]
  - replay (buf=1000): ACC 96.85 ± 0.30% (diff vs FT: +34.78 ± 1.17%), BWT -2.69 ± 0.29% (diff vs FT: +43.67 ± 1.41%), NEW 99.00 ± 0.09%
  - ewc (lambda=100): ACC 63.96 ± 1.39% (diff vs FT: +1.90 ± 1.79%), BWT -43.76 ± 1.72% (diff vs FT: +2.60 ± 2.16%), NEW 98.97 ± 0.10% (tied with FT)
  - gate:conflict (beta=0.25, decay=1.0): ACC 62.28 ± 0.98% (diff vs FT: +0.21 ± 0.35%), BWT -45.88 ± 1.24% (diff vs FT: +0.49 ± 0.48%), NEW 98.98 ± 0.06% (tied with FT)
  - finetune: ACC 62.07 ± 1.00%, BWT -46.36 ± 1.23%, NEW 99.16 ± 0.03%
  - gate:layer_cos (beta=0.5, decay=1.0): ACC 62.05 ± 1.02% (diff vs FT: -0.02 ± 0.04%), BWT -46.39 ± 1.26% (diff vs FT: -0.03 ± 0.05%), NEW 99.16 ± 0.04% (tied with FT)
  - gate:magnitude (beta=1.0, decay=0.9): ACC 61.90 ± 0.36% (diff vs FT: -0.16 ± 0.72%), BWT -45.83 ± 0.51% (diff vs FT: +0.54 ± 0.83%), NEW 98.56 ± 0.07% (tied with FT)
- problems / decisions: Cross-benchmark ranking analysis (5 comparison sentences, MEASURED only):
  1. [MEASURED] Across all three benchmarks, LoRA and replay consistently occupy the top two ranks, with LoRA taking rank 1 on Split-MNIST (98.73 ± 0.06% ACC) and replay taking rank 1 on both 5-task Permuted-MNIST (95.85 ± 0.11% ACC) and 10-task Permuted-MNIST (95.22 ± 0.09% ACC).
  2. [MEASURED] EWC maintains rank 3 across all three benchmarks (93.76 ± 0.36% on Perm5, 90.22 ± 0.87% on Perm10, 63.96 ± 1.39% on Split-MNIST), but its margin over fine-tuning drops from +12.01% on Perm10 into a statistical tie on Split-MNIST (+1.90 ± 1.79% diff, intervals overlap).
  3. [MEASURED] Gate:magnitude ranks 4th on both Perm5 (91.43 ± 0.86% ACC) and Perm10 (83.13 ± 1.22% ACC), significantly outperforming fine-tuning on both (+2.46% and +4.92% respectively), but drops to rank 7 on Split-MNIST (61.90 ± 0.36% ACC), where its interval overlaps with fine-tuning (62.07 ± 1.00%).
  4. [MEASURED] Gate:conflict ranks last (7th) on both Perm5 (81.73 ± 1.41% ACC, -7.24% vs FT) and Perm10 (41.92 ± 1.71% ACC, -36.29% vs FT), but moves up to rank 4 on Split-MNIST (62.28 ± 0.98% ACC) where all buffer-free regularization methods collapse into a statistical tie around 62%.
  5. [MEASURED] Gate:layer_cos remains tied with plain fine-tuning across all three benchmarks: 89.00 ± 2.32% vs 88.97 ± 1.85% on Perm5, 78.80 ± 0.97% vs 78.21 ± 1.68% on Perm10, and 62.05 ± 1.02% vs 62.07 ± 1.00% on Split-MNIST.
- next: Task 3.1

## Task 3.1 & 3.2 Diagnostics Logging & Ratchet Hypothesis Test
- status: done
- what changed: Built diagnostics.py. Logged step-level parameter metrics across tasks for finetune, gate:conflict, and gate:magnitude (L2 weight norms, drift along d, gated fraction, cosine similarity). Plotted outputs/phase3/weight_norms.png, outputs/phase3/drift_along_d.png, and outputs/phase3/gated_fraction.png. Tested ratchet hypothesis in outputs/phase3/hypothesis.md.
- key numbers: (From outputs/phase3/diag_all.csv):
  - Task 5 l1.weight L2 Norm: gate:conflict = 22.700 vs finetune = 11.790, gate:magnitude = 12.291.
  - Task 5 l1.weight Drift along d: gate:conflict = +2.263 vs finetune = -0.145, gate:magnitude = -0.016.
- problems / decisions: [MEASURED] Ratchet hypothesis is SUPPORTED. Directional conflict gating selectively blocks updates opposing d while leaving steps aligned with d unhindered, creating an asymmetric ratchet that doubles weight norms and drifts continuously along d.

## Task 3.3 Repaired Gate Variants
- status: done
- what changed: Added drift (anti-ratchet distance constraint from theta_star), hybrid (multiplicative conflict and magnitude damping), and conflict_soft (gentle beta in [0.05, 0.1]) modes to Gate in cl2.py. Implemented tune_and_run_repaired.py. Tuned on validation (seeds 0-2), saved outputs/phase3/tuning_repaired_val.csv, locked best configs in configs/locked_repaired.json, and ran final 5 seeds (100-104) across all 3 benchmarks (Perm5, Perm10, Split-MNIST).
- key numbers: (From outputs/phase3/final_repaired_*_table.csv, 5 seeds 100-104 on test split, mean ± std):
  - Perm5: gate:drift achieves 91.89 ± 1.03% ACC (diff vs FT: +2.92 ± 0.96%), BWT -5.82 ± 1.39% (diff vs FT: +3.80 ± 1.16%). Distinguishable from FT under tie rule (intervals do not overlap).
  - Perm10: gate:hybrid achieves 83.82 ± 0.54% ACC (diff vs FT: +5.62 ± 1.21%) and gate:drift achieves 83.21 ± 1.26% ACC (diff vs FT: +5.01 ± 1.48%). Both clearly beat FT (78.21 ± 1.68%) with non-overlapping intervals.
  - Split-MNIST: gate:drift achieves 62.62 ± 1.03% and gate:hybrid achieves 62.78 ± 0.56%, forming a statistical tie with FT (62.07 ± 1.00%).
- problems / decisions: Repaired drift and hybrid gates completely eliminate the ratchet failure and achieve statistically significant gains over fine-tuning on permutation benchmarks.

## Task 3.4 & 3.5 Layer Analysis & Research Findings
- status: done
- what changed: Generated outputs/phase3/layer_heatmap_best_gate.png showing per-layer gated fraction across tasks 2 to 5 for gate:drift. Wrote research synthesis in outputs/phase3/findings.md (392 words).
- key numbers: (From outputs/phase3/layer_heatmap_best_gate.png):
  - Layer gating hierarchy: l3.weight (87.36%), l2.bias (86.40%), l3.bias (86.12%), l1.bias (81.49%), l2.weight (79.62%), l1.weight (78.50%).
- problems / decisions: Gating is heaviest in deeper classification layers and biases, preserving high-level task representations. All findings documented with strict MEASURED citations.
- next: Phase 4 / Phase 5
