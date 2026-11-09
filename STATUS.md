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
