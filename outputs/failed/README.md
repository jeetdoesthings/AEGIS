# Failed / invalidated runs

Runs kept for the record only. None of these appear in any results table.

- `results_ewc_test_diverged.json`, `config_ewc_test_diverged.json`,
  `forgetting_curve_ewc_test_diverged.png` — EWC at the original lambda=5000 diverged
  to NaN during task 2 (diagnosis: `outputs/phase1/ewc_diagnosis.txt`). The divergence
  guard now halts such runs and flags `diverged: true` instead of reporting a number.
