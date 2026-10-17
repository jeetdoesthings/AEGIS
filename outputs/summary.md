The gradient-conflict gate did not win the benchmark. Across all tested configurations, experience replay achieved the best final accuracy at 95.2 ± 0.2% ACC (main). LoRA demonstrated the least catastrophic forgetting, maintaining 0.0 ± 0.0% BWT. Plain fine-tuning achieved the best new-task learning at 97.5 ± 0.0% NEW (epochs5).

Comparing gate variants, gate:conflict did not beat gate:magnitude; gate:magnitude:1.0 achieved 92.3 ± 0.6% ACC with -3.7 ± 0.6% BWT, severely outperforming gate:conflict:1.0 (13.7 ± 3.0% ACC, -56.9 ± 2.4% BWT).

Comparing the best setting of each baseline: plain fine-tuning (90.8 ± 2.4% ACC, -7.7 ± 2.9% BWT) decisively beat the best conflict gate setting, gate:conflict:0.5 (46.1 ± 2.1% ACC, -57.6 ± 3.0% BWT), by 44.7% accuracy. Gate:conflict beat EWC on accuracy, as EWC diverged across all evaluated lambda values (1000, 5000, 10000) to chance-level 9.8 ± 0.0% ACC.

All hyperparameters were chosen using the same test set with an identical tuning budget for EWC and the gate.
