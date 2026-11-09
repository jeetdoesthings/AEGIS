# Phase 3 Research Findings: Diagnostics and Gate Repairs

## What Was Tested
We investigated why the baseline conflict gate failed and evaluated three repaired variants across 5 seeds on test splits: `gate:drift` (adding an anti-ratchet penalty against expanding distance from start-of-task weights $\theta^*$), `gate:hybrid` (multiplying directional conflict and magnitude damping), and `gate:conflict_soft` (softening $\beta \in [0.05, 0.1]$). Step-level diagnostics tracked weight L2 norms, drift along $d$, gated fractions, and cosine similarities (`outputs/phase3/diag_all.csv`).

## What the Diagnostics Show
`[MEASURED]` The ratchet hypothesis was confirmed (`outputs/phase3/hypothesis.md`): on Task 5, `gate:conflict` drove `l1.weight` L2 norm to 22.700 (vs 11.790 for fine-tuning and 12.291 for `gate:magnitude`) and caused positive directional drift along $d$ of +2.263 (vs -0.145 for fine-tuning and -0.016 for `gate:magnitude`). Because the original conflict gate only penalized opposing steps ($g \cdot d > 0$), steps aligned with $d$ were unconstrained, causing cumulative directional runaway.

## What the Numbers Show & Distinguishability from Fine-Tuning
Applying the interval overlap tie rule ($[\text{mean} - \text{std}, \text{mean} + \text{std}]$) and paired differences:

1. **Permuted-MNIST 5-Task** (`outputs/phase3/final_repaired_perm5_table.csv`):
   - `[MEASURED]` `gate:drift` achieved 91.89 ± 1.03% ACC vs fine-tuning's 88.97 ± 1.85% (intervals $[90.86, 92.92]$ vs $[87.12, 90.82]$, no overlap). `gate:drift` is statistically distinguishable from fine-tuning, gaining +2.92 ± 0.96% paired ACC and reducing forgetting from -9.62% to -5.82% BWT (+3.80 ± 1.16% paired $\Delta$BWT).
   - `[MEASURED]` `gate:hybrid` (90.98 ± 0.90%) and `gate:conflict_soft` (90.68 ± 0.58%) overlap with fine-tuning, forming statistical ties.

2. **Permuted-MNIST 10-Task** (`outputs/phase3/final_repaired_perm10_table.csv`):
   - `[MEASURED]` Both `gate:hybrid` (83.82 ± 0.54% ACC, $[83.28, 84.36]$) and `gate:drift` (83.21 ± 1.26% ACC, $[81.95, 84.47]$) are statistically distinguishable from fine-tuning (78.21 ± 1.68% ACC, $[76.53, 79.89]$), outperforming it by +5.62 ± 1.21% and +5.01 ± 1.48% paired ACC, and reducing forgetting by +6.84 ± 1.30% and +5.82 ± 1.59% $\Delta$BWT.

3. **Split-MNIST** (`outputs/phase3/final_repaired_splitmnist_table.csv`):
   - `[MEASURED]` All gate variants overlap with fine-tuning (62.07 ± 1.00% ACC vs 62.62 ± 1.03% for drift, 62.78 ± 0.56% for hybrid), representing statistical ties under severe output-head class interference.

4. **Layer Gating Hierarchy** (`outputs/phase3/layer_heatmap_best_gate.png`):
   - `[MEASURED]` In `gate:drift`, gating fraction was highest on `l3.weight` (87.36%), `l2.bias` (86.40%), and `l3.bias` (86.12%), and lowest on `l1.weight` (78.50%).

## What Remains Unexplained
`[HYPOTHESIS]` While adding the drift term successfully bounds weight norms and reverses catastrophic failure on permutation benchmarks, buffer-free gradient gating cannot resolve domain-incremental class interference in Split-MNIST, where exemplars (`replay`) or separate heads/adapters (`lora`) remain necessary.
