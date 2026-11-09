# Phase 5 Figure & Table Captions

This document contains standardized, publication-ready captions for every visual deliverable and quantitative summary in `outputs/phase5/`. Every caption specifies the benchmark, dataset, evaluation split, number of test seeds, metric definition, and explicit note on uncertainty representation (mean ± std).

---

### Table 1: Permuted-MNIST 5-Task Quantitative Evaluation
- **Files**: [`outputs/phase5/table_perm5.csv`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/table_perm5.csv), [`outputs/phase5/table_perm5.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/table_perm5.png)
- **Caption**: *Table 1: Continual learning performance on 5-task Permuted-MNIST evaluated across 5 fixed random seeds (100, 101, 102, 103, 104) on the held-out test split (10,000 samples per task). All values are reported as mean ± std across seeds. ACC denotes final average accuracy across all 5 tasks; BWT denotes backward transfer (accuracy change on earlier tasks after subsequent training); FWT denotes forward transfer relative to an untrained network; NEW denotes accuracy on each task immediately following its learning. Δ ACC vs FT and Δ BWT vs FT represent per-seed paired differences relative to plain fine-tuning. Blue bold indicates top ACC; green bold indicates top BWT. \*LoRA utilizes an oracle task ID at test time to route representations to task-specific adapters.*

---

### Table 2: Permuted-MNIST 10-Task Quantitative Evaluation
- **Files**: [`outputs/phase5/table_perm10.csv`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/table_perm10.csv), [`outputs/phase5/table_perm10.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/table_perm10.png)
- **Caption**: *Table 2: Continual learning performance on 10-task sequential Permuted-MNIST evaluated across 5 fixed random seeds (100, 101, 102, 103, 104) on the held-out test split. All values are reported as mean ± std. Over 10 tasks, plain fine-tuning drops to 78.21 ± 1.68% ACC (BWT -20.80 ± 1.84%), while repaired anti-ratchet gates (gate:hybrid at 83.82 ± 0.54% and gate:drift at 83.21 ± 1.26%) statistically outperform fine-tuning by over +5% ACC without memory replay buffers. \*LoRA utilizes an oracle task ID at test time.*

---

### Table 3: Split-MNIST Domain-Incremental 5-Task Quantitative Evaluation
- **Files**: [`outputs/phase5/table_split.csv`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/table_split.csv), [`outputs/phase5/table_split.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/table_split.png)
- **Caption**: *Table 3: Continual learning performance on domain-incremental Split-MNIST (5 sequential binary classification tasks on digits 0-9 sharing a 2-way output head) across 5 test seeds (100-104) on the held-out test split. Values reported as mean ± std. In this setting, all buffer-free weight regularization and gating methods collapse into a statistical tie around 62% ACC due to severe head interference, while exemplar replay (96.85 ± 0.30%) and parameter-isolated LoRA (98.73 ± 0.06%) maintain high performance.*

---

### Table 4: Master Cross-Benchmark Summary
- **Files**: [`outputs/phase5/master_summary.csv`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/master_summary.csv), [`outputs/phase5/master_summary.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/master_summary.png)
- **Caption**: *Table 4: Comprehensive continual learning comparison across all three experimental benchmarks (Permuted-MNIST 5-task, Permuted-MNIST 10-task, Split-MNIST 5-task) evaluated across 5 random seeds (100-104) on held-out test splits. Metrics shown are final Average Accuracy (ACC, %) and Backward Transfer (BWT, %), formatted as mean ± std.*

---

### Figure 1: Task 1 Accuracy Retention Curve (Permuted-MNIST 5 Tasks)
- **Files**: [`outputs/phase5/forgetting_curve_perm5.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/forgetting_curve_perm5.png), [`outputs/phase5/forgetting_curve_perm5_all.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/forgetting_curve_perm5_all.png)
- **Caption**: *Figure 1: Task 1 test accuracy retention ($R_{i, 1}$) as a function of subsequent tasks learned ($i = 1, \dots, 5$) on Permuted-MNIST across 5 test seeds (100-104). Solid curves represent the mean accuracy; shaded bands represent ±1 standard deviation across seeds. Notice that gate:drift (cyan) and gate:hybrid (olive) maintain Task 1 accuracy substantially higher than plain fine-tuning (blue), while original gate:conflict (purple dash-dot) suffers catastrophic runaway forgetting.*

---

### Figure 2: Task 1 Accuracy Retention Curve (Permuted-MNIST 10 Tasks)
- **Files**: [`outputs/phase5/forgetting_curve_perm10.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/forgetting_curve_perm10.png), [`outputs/phase5/forgetting_curve_perm10_all.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/forgetting_curve_perm10_all.png)
- **Caption**: *Figure 2: Task 1 test accuracy retention ($R_{i, 1}$) across 10 sequential tasks on Permuted-MNIST (5 seeds: 100-104). Solid/dashed curves show mean; shaded bands show ±1 std. On long task sequences, original gate:conflict degrades to near-chance ($13.44 \pm 4.44\%$ on Task 1), while gate:magnitude preserves $65.20 \pm 7.04\%$, gate:drift preserves $54.44 \pm 5.51\%$, and gate:hybrid preserves $52.71 \pm 4.93\%$ Task 1 accuracy, compared to $47.31 \pm 4.80\%$ for plain fine-tuning.*

---

### Figure 3: Task 1 Accuracy Retention Curve (Split-MNIST)
- **Files**: [`outputs/phase5/forgetting_curve_split.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/forgetting_curve_split.png)
- **Caption**: *Figure 3: Task 1 test accuracy retention ($R_{i, 1}$) across 5 sequential digit-pair tasks on domain-incremental Split-MNIST (5 seeds: 100-104, mean ± std). Fine-tuning and all buffer-free gating methods immediately plunge from ~99% to ~50% (random guess between the two active binary classes) upon training on Task 2, whereas replay and LoRA retain near 100% accuracy.*

---

### Figure 4: Test Accuracy Matrix Heatmaps (Permuted-MNIST 5 Tasks)
- **Files**: [`outputs/phase5/heatmaps_perm5.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/heatmaps_perm5.png)
- **Caption**: *Figure 4: Final accuracy matrix $R \in \mathbb{R}^{5 \times 5}$ averaged over 5 test seeds (100-104) on Permuted-MNIST for seven continual learning methods. Matrix entry $R_{i, j}$ represents test accuracy on task $j$ after learning task $i$. The diagonal represents immediate task performance ($NEW$); lower-triangular entries show backward transfer and catastrophic forgetting on previously learned tasks.*

---

### Figure 5: Test Accuracy Matrix Heatmaps (Permuted-MNIST 10 Tasks)
- **Files**: [`outputs/phase5/heatmaps_perm10.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/heatmaps_perm10.png)
- **Caption**: *Figure 5: Final accuracy matrix $R \in \mathbb{R}^{10 \times 10}$ averaged over 5 test seeds (100-104) on 10-task Permuted-MNIST. Shows progressive degradation across 10 sequential tasks for fine-tuning, EWC, replay, LoRA, and original gating methods.*

---

### Figure 6: Test Accuracy Matrix Heatmaps (Split-MNIST)
- **Files**: [`outputs/phase5/heatmaps_split.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/heatmaps_split.png)
- **Caption**: *Figure 6: Final accuracy matrix $R \in \mathbb{R}^{5 \times 5}$ averaged over 5 test seeds (100-104) on domain-incremental Split-MNIST. Highlights extreme catastrophic forgetting across binary tasks for shared-head architectures lacking replay buffers.*

---

### Figure 7: Mechanistic Diagnostics — Parameter L2 Weight Norms
- **Files**: [`outputs/phase5/diagnostics_weight_norms.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/diagnostics_weight_norms.png)
- **Caption**: *Figure 7: Step-by-step parameter L2 norms ($\|\theta_l\|_2$) across 5 Permuted-MNIST tasks for fine-tuning, gate:conflict, and gate:magnitude. Gate:conflict exhibits explosive parameter growth in layer 1 weights ($l1.weight$ norm reaches 22.700 by Task 5, compared to 11.790 for fine-tuning), confirming the runaway ratchet mechanism.*

---

### Figure 8: Mechanistic Diagnostics — Cumulative Parameter Drift Along Accumulated Direction $d$
- **Files**: [`outputs/phase5/diagnostics_drift_along_d.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/diagnostics_drift_along_d.png)
- **Caption**: *Figure 8: Parameter displacement projected onto the accumulated drift direction ($\frac{(\theta - \theta_{start}) \cdot d}{\|d\|_2}$) across training steps. In gate:conflict, blocking opposing gradient updates while permitting aligned updates causes unidirectional runaway drift (+2.263 on Task 5 $l1.weight$, versus -0.145 for fine-tuning and -0.016 for gate:magnitude).*

---

### Figure 9: Mechanistic Diagnostics — Fraction of Gated Parameters
- **Files**: [`outputs/phase5/diagnostics_gated_fraction.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/diagnostics_gated_fraction.png)
- **Caption**: *Figure 9: Fraction of parameter elements receiving non-zero damping scale per optimization step across tasks 2 to 5 for gate:conflict and gate:magnitude. More than 80% of parameters trigger conflict damping at every step.*

---

### Figure 10: Layer Gating Hierarchy Heatmap
- **Files**: [`outputs/phase5/layer_heatmap_best_gate.png`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/layer_heatmap_best_gate.png)
- **Caption**: *Figure 10: Proportion of weights gated (%) per layer and task for the best repaired gate (gate:drift:0.1) across tasks 2 through 5. Classification head layers ($l3.weight$: 87.36%, $l3.bias$: 86.12%) undergo substantially heavier gating than early feature extraction layers ($l1.weight$: 78.50%), showing that preservation naturally concentrates where semantic representations are specialized.*
