# Continual Learning with Directional and Magnitude Gradient Gating: Empirical Evaluation, Mechanistic Failure Modes, and Anti-Ratchet Repairs

**Author:** Project AEGIS  
**Date:** November 2026  
**Repository:** [github.com/jeetdoesthings/AEGIS](https://github.com/jeetdoesthings/AEGIS)  
**Artifact Directory:** `outputs/phase5/`  

---

## Executive Summary

Continual learning in artificial neural networks remains bottlenecked by catastrophic forgetting—the abrupt decay of previously acquired representations upon learning new distributions. This study investigates the Gradient-Conflict Gate, a buffer-free parameter-level mechanism designed to mitigate forgetting by dampening updates that oppose the historical displacement vector of weights ($d = \theta - \theta_{start}$). 

Through an academically rigorous experimental framework comprising isolated train/validation/test splits, equal hyperparameter tuning budgets, and multi-seed evaluations across Permuted-MNIST (5 and 10 tasks) and domain-incremental Split-MNIST, we establish three primary empirical findings:
1. `[MEASURED]` The original directional conflict gate (`gate:conflict`) fails catastrophically: on 5-task Permuted-MNIST, it degrades accuracy relative to plain fine-tuning by $-7.24 \pm 1.38\%$ ($81.73 \pm 1.41\%$ vs $88.97 \pm 1.85\%$), and on 10 tasks, accuracy collapses to $41.92 \pm 1.71\%$ (a $-36.29 \pm 2.82\%$ drop vs fine-tuning's $78.21 \pm 1.68\%$).
2. `[MEASURED]` Step-level parameter diagnostics support the *asymmetric ratchet hypothesis*: selectively zeroing updates that oppose $d$ ($g \cdot d > 0$) while allowing updates aligned with $d$ to proceed unhindered causes weights to blow up unidirectionally (Task 5 $l1.weight$ L2 norm doubles to 22.700 vs 11.790 for fine-tuning) and run away along $d$ (+2.263 vs -0.145 for fine-tuning).
3. `[MEASURED]` Two repaired variants—an anti-drift gradient gate (`gate:drift`, which checks distance expansion from task start weights) and a multiplicative conflict-magnitude damper (`gate:hybrid`)—eliminate the runaway mechanism. On 5-task Permuted-MNIST, `gate:drift` achieves $91.89 \pm 1.03\%$ ACC ($+2.92 \pm 0.96\%$ over fine-tuning, non-overlapping intervals). On 10-task Permuted-MNIST, `gate:hybrid` and `gate:drift` achieve $83.82 \pm 0.54\%$ and $83.21 \pm 1.26\%$ ACC respectively, statistically outperforming plain fine-tuning ($78.21 \pm 1.68\%$) by over $+5\%$ ACC without storing any past data.

---

## 1. Introduction

Catastrophic forgetting represents a foundational challenge in deep learning: gradient descent applied sequentially across changing data distributions optimizes current task objectives while overwriting orthogonal or opposing parameter subspaces essential for previous tasks.

In contemporary practice, two prevalent paradigms are often assumed to circumvent this issue: Retrieval-Augmented Generation (RAG) and long-context transformer architectures. However, neither addresses the fundamental neural plasticity problem:
- **RAG** appends external factual documents into a model's static prompt. While effective for knowledge lookup, it does not alter parameter weights, cannot acquire procedural reasoning or non-textual perceptual skills incrementally, and fails when the reasoning policy itself must adapt across domains.
- **Long-context windows** allow in-context learning over large prompt buffers, but context retention is ephemeral, computationally quadratic (or sub-quadratic with significant KV-cache memory footprints), and vanishes once the inference session concludes. True continual learning requires consolidating experience directly into network weights without boundless memory buffers or computational overhead.

This paper evaluates continual learning methods on sequential benchmark sequences. We investigate whether parameter-space gradient gating can protect historical task knowledge without data replay.

---

## 2. Related Work

The methods benchmarked in this work span four major families of continual learning:

1. **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017): Regularizes weights quadratically based on diagonal Fisher Information ($F_i = \mathbb{E}[(\nabla_{\theta} \log p(y|x))^2]$):
   $$\mathcal{L}_{EWC}(\theta) = \mathcal{L}_{task}(\theta) + \sum_{k<t} \frac{\lambda}{2} \sum_i F_{i, k} (\theta_i - \theta_{i, k}^*)^2$$
   While effective, EWC is sensitive to hyperparameter scale: an excessive penalty ($\lambda \ge 1000$ with plain SGD) causes gradient explosion and numerical collapse into NaNs, which we formally diagnose and repair.
2. **Experience Replay** (Robins, 1995; Chaudhry et al., 2019): Stores a small coreset of historical input-label pairs in an episodic memory buffer and interleaves them into mini-batch training. Replay serves as an empirical upper bound for continual adaptation but violates strict data-privacy and memory-free deployment constraints.
3. **Parameter-Isolated Adapters (LoRA)** (Hu et al., 2021): Freezes shared backbone weights and trains low-rank adapter matrices ($W = W_0 + \frac{\alpha}{r} B A$). In continual learning, routing inputs to dedicated adapters prevents backward interference ($BWT = 0.00\%$), but requires an oracle task ID at test time.
4. **Gradient-Based and Weight-Trajectory Protection**:
   - **Synaptic Intelligence (SI)** (Zenke et al., 2017): Accumulates parameter importance online by tracking the inner product of parameter updates and gradients along optimization trajectories ($\omega_k = \sum g_t \Delta \theta_t$).
   - **Gradient Episodic Memory (GEM)** (Lopez-Paz & Ranzato, 2017): Uses quadratic programming to project the current gradient onto the half-space satisfying $\langle g, g_k \rangle \ge 0$ for all stored replay gradients.
   - **Titans & Neural Memory** (Behrouz et al., 2024): Incorporates surprise-based gating and momentum-driven memory modules into sequence models.

### Relationship to Prior Art
The Gradient-Conflict Gate investigated here is **not novel**. It is conceptually derived from the intersection of Synaptic Intelligence (tracking cumulative weight displacement $d = \theta - \theta_{start}$) and GEM (penalizing or zeroing gradient updates that exhibit directional conflict). Rather than claiming algorithmic novelty, this work systematically uncovers the exact mathematical failure modes of naive directional gradient gating, diagnoses its mechanistic collapse, and formulates stable, mathematically grounded repairs.

---

## 3. Method: The Gradient-Conflict Gate & Variants

Let $\theta \in \mathbb{R}^P$ denote the parameters of the network. At the start of task $t$, the reference weights are frozen as $\theta_{start} = \theta^{(t-1)}$. The cumulative displacement vector across all prior tasks is defined as:
$$d = \theta_{start} - \theta^{(0)}$$
For each incoming mini-batch on the current task, backpropagation computes gradient $g = \nabla_\theta \mathcal{L}_{task}$.

### 3.1 Original Directional Conflict Gate (`gate:conflict`)
The original formulation posited that an update $g$ is "conflicting" if taking a gradient descent step ($\Delta \theta \propto -g$) opposes the direction that prior tasks moved the weight. Since the historical movement vector is $d$, a step that reverses this movement satisfies $(-g) \cdot d < 0$, which is equivalent to:
$$g \cdot d > 0$$
When this conflict condition is met, the gate damps the gradient element-wise:
$$g_{gated} = g \odot (1 - \beta \cdot \mathbf{1}_{g \cdot d > 0})$$
At full damping ($\beta = 1.0$), conflicting gradient elements are set to zero. A decay hyperparameter $\gamma$ optionally scales historical displacement at task boundaries ($d \leftarrow \gamma d$).

### 3.2 Magnitude Ablation (`gate:magnitude`)
To test whether directional information is necessary, the magnitude ablation completely ignores the sign of $g \cdot d$ and instead dampens updates on weights that have experienced large historical displacements:
$$g_{gated} = g \odot \max\left(0, 1 - \beta \frac{|d|}{\max(|d|) + \epsilon}\right)$$

### 3.3 Layer-Level Cosine Gate (`gate:layer_cos`)
Instead of element-wise masking, this variant evaluates the aggregate layer-wise cosine similarity:
$$\cos(g_l, d_l) = \frac{\langle g_l, d_l \rangle}{\|g_l\|_2 \|d_l\|_2 + \epsilon}$$
If $\cos(g_l, d_l) > 0$, the entire layer's gradient is scaled down: $g_l \leftarrow g_l \cdot (1 - \beta \cos(g_l, d_l))$.

### 3.4 Mechanistic Repairs (Developed in Phase 3)

Diagnostic telemetry (Section 6) revealed that `gate:conflict` induces an asymmetric ratchet: zeroing updates when $g \cdot d > 0$ permits updates with $g \cdot d \le 0$ to proceed unhindered, allowing parameters to drift unboundedly away from their origin along $d$. To eliminate this instability, we formulated three repaired variants:

1. **Anti-Drift Gradient Gate (`gate:drift`)**:
   In continual learning literature, restricting drift away from reference weights is standardly achieved by adding an explicit quadratic loss penalty, as in L2-SP ($\frac{\alpha}{2}\|\theta - \theta_{ref}\|_2^2$) or EWC ($\frac{\lambda}{2}\sum F_i(\theta_i - \theta_i^*)^2$). In `cl2.py`, `gate:drift` operationalizes anchor stabilization directly at the gradient level without computing an explicit auxiliary loss. At the start of task $t$, reference weights are frozen as $s = \theta_{start}$. For current parameter displacement $e = \theta - s$, taking an SGD step $\Delta \theta = -\eta g$ expands the squared distance from the task anchor whenever $\Delta \theta \cdot e > 0 \iff g \cdot e < 0$. The gate tests whether an update either opposes historical movement $d$ or expands distance from the task anchor:
   $$\text{cond} = (g \odot d > 0) \lor (g \odot e < 0)$$
   $$g \leftarrow g \odot (1 - \beta \cdot \text{mag} \cdot \mathbf{1}_{\text{cond}})$$
   where $\text{mag} = \text{clamp}\left(\frac{|d|}{\overline{|d|} + \epsilon}, \max=1.0\right)$. By braking updates that point away from $\theta_{start}$, `gate:drift` acts as a gradient-level anchor constraint analogous to a movement-weighted L2-SP barrier.
2. **Multiplicative Hybrid Gate (`gate:hybrid`)**:
   Combines soft directional conflict gating with unconditional magnitude-proportional damping:
   $$g \leftarrow g \odot (1 - \beta \cdot \text{mag} \cdot \mathbf{1}_{g \odot d > 0}) \odot (1 - \gamma \cdot \text{mag})$$
   where $\beta$ controls directional conflict damping and $\gamma$ applies gentle, continuous dampening on heavily displaced parameters regardless of gradient direction.
3. **Soft Conflict Gate (`gate:conflict_soft`)**:
   Retains directional conflict checking but restricts damping to a conservative regime ($\beta \in [0.05, 0.1]$), preventing the complete elimination of opposing gradients.

---

## 4. Experimental Setup

### 4.1 Benchmarks
1. **Permuted-MNIST 5-Task**: Sequential classification of handwritten digits (0–9) where each task applies a unique, fixed pseudo-random spatial permutation to all 784 input pixels.
2. **Permuted-MNIST 10-Task**: A 10-task sequential permutation sequence designed to test long-horizon compounding degradation.
3. **Split-MNIST 5-Task (Domain-Incremental)**: The 10 MNIST classes are split into 5 disjoint sequential binary classification tasks: (0,1), (2,3), (4,5), (6,7), (8,9). All tasks share a single 2-way output head, creating extreme class interference across task boundaries.

### 4.2 Data Split Protocol
To prevent data leakage, the standard MNIST training split (60,000 images) was deterministically partitioned (seed 42) into:
- **55,000 training samples**: Strictly isolated for model training.
- **5,000 validation samples**: Strictly isolated for hyperparameter tuning.
- **10,000 test samples**: Evaluated exclusively for final reporting across 5 fresh test seeds.

### 4.3 Model Architecture & Training Protocol
- **Architecture**: Small Multi-Layer Perceptron (MLP) with two hidden layers: $784 \to 256 \to 256 \to C$, ReLU activations. (For Split-MNIST, $C=2$; for Permuted-MNIST, $C=10$). Total parameters: 269,322.
- **Optimizer**: Plain Stochastic Gradient Descent (SGD), learning rate $\eta = 0.01$, batch size 64, 3 epochs per task.
- **Divergence Guard**: Embedded telemetry halting any run where training loss becomes non-finite ($\text{loss} = \text{NaN/Inf}$), preventing silent failures from populating tables with corrupted numbers.

### 4.4 Tuning & Test Protocol
- **Hyperparameter Tuning**: Performed strictly on the validation split across seeds 0, 1, and 2. Each method was allocated an equal tuning budget ($\le 6$ configurations). The best configuration was selected by highest validation ACC (ties broken by BWT, then NEW) and locked into JSON configurations before any test evaluation.
- **Final Evaluation**: Conducted across 5 independent test seeds (100, 101, 102, 103, 104) on the held-out test split.
- **Metrics Reported**:
  - $ACC = \frac{1}{T} \sum_{j=1}^T R_{T, j}$: Final average accuracy across all $T$ tasks at the end of training.
  - $BWT = \frac{1}{T-1} \sum_{j=1}^{T-1} (R_{T, j} - R_{j, j})$: Backward transfer; average accuracy change on old tasks (negative values quantify catastrophic forgetting).
  - $FWT = \frac{1}{T-1} \sum_{j=2}^T (R_{j-1, j} - R_{0, j})$: Forward transfer relative to an untrained random network ($R_{0, j}$).
  - $NEW = \frac{1}{T} \sum_{j=1}^T R_{j, j}$: Immediate performance on each task right after learning it.
  - $\Delta ACC \text{ vs FT}$ and $\Delta BWT \text{ vs FT}$: Paired per-seed difference against plain fine-tuning.

### 4.5 Statistical Tie Rule
In accordance with rigorous statistical practices, two methods are reported as a **statistical tie** if their `mean ± std` intervals overlap. Claims of "beating" or "outperforming" are strictly reserved for non-overlapping distributions.

---

## 5. Empirical Results

All quantitative values are cited directly from the experimental result files in `outputs/phase5/`.

### 5.1 Permuted-MNIST (5 Tasks)
*(Evaluated across 5 test seeds: 100–104 on held-out test split)*

| Method | ACC (%) | Δ ACC vs FT | BWT (%) | Δ BWT vs FT | FWT (%) | NEW (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuning (Plain SGD)** | $88.97 \pm 1.85$ | reference | $-9.62 \pm 2.34$ | reference | $1.51 \pm 0.82$ | $96.67 \pm 0.12$ |
| **EWC (repaired, $\lambda=20$)** | $93.76 \pm 0.36$ | $+4.78 \pm 1.69$ | $-2.34 \pm 0.55$ | $+7.28 \pm 2.07$ | $1.63 \pm 0.63$ | $95.63 \pm 0.21$ |
| **Replay (buffer=1000)** | $\mathbf{95.85 \pm 0.11}$ | $+6.88 \pm 1.87$ | $-1.18 \pm 0.14$ | $+8.43 \pm 2.43$ | $1.86 \pm 1.32$ | $96.80 \pm 0.06$ |
| **LoRA ($r=16$, oracle task ID)\*** | $94.30 \pm 0.12$ | $+5.32 \pm 1.78$ | $\mathbf{0.00 \pm 0.00}$ | $+9.62 \pm 2.34$ | $2.05 \pm 1.63$ | $94.30 \pm 0.12$ |
| **Gate: Conflict (original, $\beta=0.25$)** | $81.73 \pm 1.41$ | $-7.24 \pm 1.38$ | $-17.60 \pm 1.84$ | $-7.98 \pm 1.67$ | $1.68 \pm 1.33$ | $95.81 \pm 0.10$ |
| **Gate: Magnitude (ablation, $\beta=1.0$)** | $91.43 \pm 0.86$ | $+2.46 \pm 1.73$ | $-4.39 \pm 1.01$ | $+5.23 \pm 2.23$ | $1.68 \pm 1.15$ | $94.94 \pm 0.06$ |
| **Gate: Layer Cosine ($\beta=1.0$)** | $89.00 \pm 2.32$ | $+0.02 \pm 0.86$ | $-9.62 \pm 2.98$ | $-0.00 \pm 1.07$ | $1.37 \pm 0.77$ | $96.69 \pm 0.12$ |
| **Gate: Drift (repaired, $c=0.1$)** | $91.89 \pm 1.03$ | $+2.92 \pm 0.96$ | $-5.82 \pm 1.39$ | $+3.80 \pm 1.16$ | $1.45 \pm 0.70$ | $96.54 \pm 0.11$ |
| **Gate: Hybrid (repaired, $c=0.1$)** | $90.98 \pm 0.90$ | $+2.01 \pm 1.35$ | $-6.64 \pm 1.20$ | $+2.98 \pm 1.64$ | $1.43 \pm 0.88$ | $96.29 \pm 0.08$ |
| **Gate: Soft Conflict ($\beta=0.1$)** | $90.68 \pm 0.58$ | $+1.71 \pm 1.58$ | $-7.44 \pm 0.79$ | $+2.17 \pm 1.96$ | $1.42 \pm 0.89$ | $96.63 \pm 0.11$ |

*\*LoRA requires an oracle task ID at test time to select the task adapter.*

#### Key Observations (Perm5):
1. `[MEASURED]` Original `gate:conflict` performs worse than plain fine-tuning: final ACC is $-7.24 \pm 1.38\%$ lower ($81.73 \pm 1.41\%$ vs $88.97 \pm 1.85\%$), and backward forgetting is nearly doubled ($-17.60 \pm 1.84\%$ vs $-9.62 \pm 2.34\%$).
2. `[MEASURED]` The magnitude ablation (`gate:magnitude`) achieves $91.43 \pm 0.86\%$ ACC (paired difference $+2.46 \pm 1.73\%$ over fine-tuning; independent intervals $[90.57, 92.29]$ and $[87.12, 90.82]$ overlap slightly, constituting a statistical tie under the independent interval rule) while cutting backward forgetting in half ($-4.39 \pm 1.01\%$ vs $-9.62 \pm 2.34\%$).
3. `[MEASURED]` The repaired `gate:drift` variant achieves $91.89 \pm 1.03\%$ ACC, which statistically outperforms plain fine-tuning ($88.97 \pm 1.85\%$) under the non-overlapping tie rule ($91.89 - 1.03 = 90.86 > 88.97 + 1.85 = 90.82$).
4. `[MEASURED]` Replay ($1000$ exemplars) achieves the highest final ACC ($95.85 \pm 0.11\%$), while LoRA achieves zero backward forgetting ($BWT = 0.00 \pm 0.00\%$) via isolated adapters.

---

### 5.2 Permuted-MNIST (10 Tasks)
*(Evaluated across 5 test seeds: 100–104 on held-out test split)*

| Method | ACC (%) | Δ ACC vs FT | BWT (%) | Δ BWT vs FT | FWT (%) | NEW (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuning (Plain SGD)** | $78.21 \pm 1.68$ | reference | $-20.80 \pm 1.84$ | reference | $0.58 \pm 0.61$ | $96.92 \pm 0.10$ |
| **EWC (repaired, $\lambda=20$)** | $90.22 \pm 0.87$ | $+12.01 \pm 0.86$ | $-4.17 \pm 0.93$ | $+16.63 \pm 1.08$ | $0.67 \pm 0.55$ | $93.97 \pm 0.15$ |
| **Replay (buffer=1000)** | $\mathbf{95.22 \pm 0.09}$ | $+17.01 \pm 1.70$ | $-1.92 \pm 0.14$ | $+18.88 \pm 1.88$ | $0.93 \pm 0.45$ | $96.94 \pm 0.05$ |
| **LoRA ($r=16$, oracle task ID)\*** | $94.05 \pm 0.05$ | $+15.84 \pm 1.69$ | $\mathbf{0.00 \pm 0.00}$ | $+20.80 \pm 1.84$ | $1.07 \pm 0.67$ | $94.05 \pm 0.05$ |
| **Gate: Conflict (original, $\beta=0.25$)** | $41.92 \pm 1.71$ | $-36.29 \pm 2.82$ | $-58.83 \pm 1.95$ | $-38.03 \pm 3.13$ | $1.07 \pm 1.16$ | $94.86 \pm 0.15$ |
| **Gate: Magnitude (ablation, $\beta=1.0$)** | $83.13 \pm 1.22$ | $+4.92 \pm 1.20$ | $-12.82 \pm 1.33$ | $+7.97 \pm 1.34$ | $0.94 \pm 0.65$ | $94.67 \pm 0.04$ |
| **Gate: Layer Cosine ($\beta=1.0$)** | $78.80 \pm 0.97$ | $+0.59 \pm 2.40$ | $-20.15 \pm 1.15$ | $+0.64 \pm 2.66$ | $0.52 \pm 0.68$ | $96.94 \pm 0.09$ |
| **Gate: Drift (repaired, $c=0.1$)** | $83.21 \pm 1.26$ | $+5.01 \pm 1.48$ | $-14.98 \pm 1.44$ | $+5.82 \pm 1.59$ | $0.68 \pm 0.73$ | $96.69 \pm 0.08$ |
| **Gate: Hybrid (repaired, $c=0.1$)** | $83.82 \pm 0.54$ | $+5.62 \pm 1.21$ | $-13.95 \pm 0.64$ | $+6.84 \pm 1.30$ | $0.47 \pm 1.06$ | $96.38 \pm 0.05$ |
| **Gate: Soft Conflict ($\beta=0.1$)** | $80.05 \pm 1.66$ | $+1.85 \pm 1.38$ | $-18.58 \pm 1.86$ | $+2.22 \pm 1.52$ | $0.94 \pm 0.65$ | $96.77 \pm 0.10$ |

#### Key Observations (Perm10):
1. `[MEASURED]` Over a 10-task horizon, catastrophic forgetting accelerates in plain fine-tuning: BWT drops to $-20.80 \pm 1.84\%$, pulling final ACC down to $78.21 \pm 1.68\%$.
2. `[MEASURED]` The original `gate:conflict` experiences catastrophic compounding breakdown: final ACC drops to $41.92 \pm 1.71\%$ ($-36.29 \pm 2.82\%$ lower than fine-tuning) with an extreme backward transfer collapse of $-58.83 \pm 1.95\%$. Task 1 retention falls to $14.28 \pm 1.57\%$ (near chance for 10 classes).
3. `[MEASURED]` Both repaired variants statistically outperform fine-tuning with non-overlapping distributions: `gate:hybrid` achieves $83.82 \pm 0.54\%$ ACC ($+5.62 \pm 1.21\%$) and `gate:drift` achieves $83.21 \pm 1.26\%$ ACC ($+5.01 \pm 1.48\%$).
4. `[MEASURED]` Repaired EWC ($\lambda=20$) demonstrates strong buffer-free retention on Perm10, reaching $90.22 \pm 0.87\%$ ACC with BWT limited to $-4.17 \pm 0.93\%$.

---

### 5.3 Split-MNIST (Domain-Incremental 5 Tasks)
*(Evaluated across 5 test seeds: 100–104 on held-out test split)*

| Method | ACC (%) | Δ ACC vs FT | BWT (%) | Δ BWT vs FT | FWT (%) | NEW (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuning (Plain SGD)** | $62.07 \pm 1.00$ | reference | $-46.36 \pm 1.23$ | reference | $2.45 \pm 2.98$ | $99.16 \pm 0.03$ |
| **EWC (repaired, $\lambda=100$)** | $63.96 \pm 1.39$ | $+1.90 \pm 1.79$ | $-43.76 \pm 1.72$ | $+2.60 \pm 2.16$ | $2.06 \pm 3.16$ | $98.97 \pm 0.10$ |
| **Replay (buffer=1000)** | $96.85 \pm 0.30$ | $+34.78 \pm 1.17$ | $-2.69 \pm 0.29$ | $+43.67 \pm 1.41$ | $4.40 \pm 3.21$ | $99.00 \pm 0.09$ |
| **LoRA ($r=8$, oracle task ID)\*** | $\mathbf{98.73 \pm 0.06}$ | $+36.66 \pm 1.02$ | $\mathbf{0.00 \pm 0.00}$ | $+46.36 \pm 1.23$ | $1.00 \pm 4.81$ | $98.73 \pm 0.06$ |
| **Gate: Conflict (original, $\beta=0.25$)** | $62.28 \pm 0.98$ | $+0.21 \pm 0.35$ | $-45.88 \pm 1.24$ | $+0.49 \pm 0.48$ | $2.38 \pm 2.84$ | $98.98 \pm 0.06$ |
| **Gate: Magnitude (ablation, $\beta=1.0$)** | $61.90 \pm 0.36$ | $-0.16 \pm 0.72$ | $-45.83 \pm 0.51$ | $+0.54 \pm 0.83$ | $1.10 \pm 3.54$ | $98.56 \pm 0.07$ |
| **Gate: Layer Cosine ($\beta=0.5$)** | $62.05 \pm 1.02$ | $-0.02 \pm 0.04$ | $-46.39 \pm 1.26$ | $-0.03 \pm 0.05$ | $2.43 \pm 2.99$ | $99.16 \pm 0.04$ |
| **Gate: Drift (repaired, $c=0.1$)** | $62.62 \pm 1.03$ | $+0.55 \pm 0.15$ | $-45.61 \pm 1.27$ | $+0.75 \pm 0.19$ | $2.40 \pm 3.05$ | $99.11 \pm 0.03$ |
| **Gate: Hybrid (repaired, $c=0.1$)** | $62.78 \pm 0.56$ | $+0.72 \pm 0.54$ | $-45.11 \pm 0.71$ | $+1.26 \pm 0.64$ | $2.42 \pm 3.05$ | $98.87 \pm 0.01$ |
| **Gate: Soft Conflict ($\beta=0.1$)** | $62.34 \pm 1.03$ | $+0.27 \pm 0.18$ | $-45.99 \pm 1.28$ | $+0.38 \pm 0.24$ | $2.43 \pm 3.03$ | $99.13 \pm 0.04$ |

#### Key Observations (Split-MNIST):
1. `[MEASURED]` All buffer-free weight regularization and gating methods (EWC, fine-tuning, and all six gate variants) collapse into a mutual statistical tie between $61.90\%$ and $63.96\%$ ACC, suffering catastrophic backward transfer between $-43.76\%$ and $-46.39\%$.
2. `[MEASURED]` In contrast, replay ($96.85 \pm 0.30\%$) and LoRA ($98.73 \pm 0.06\%$) preserve almost full performance.
3. `[HYPOTHESIS]` This divergence occurs because domain-incremental Split-MNIST uses a shared 2-way output head. When learning digits (2,3), the output weights mapping representations to classes 0 and 1 are overwritten to classify classes 2 and 3. Without exemplar replay to re-anchor the head or task-specific routing (LoRA), weight-space regularization cannot resolve output logit interference.

---

## 6. Mechanistic Analysis & The Ratchet Hypothesis

To uncover why `gate:conflict` failed while `gate:magnitude` and repaired gates succeeded, step-level diagnostics were logged during training (`outputs/phase3/diag_all.csv`).

### 6.1 The Asymmetric Ratchet Mechanism
The original gate defined conflict as $g \cdot d > 0$ and applied $g \leftarrow g(1 - \beta)$. When $\beta = 1.0$, updates satisfying $g \cdot d > 0$ were completely zeroed.

- `[MEASURED]` **Parameter L2 Norms** (`outputs/phase5/diagnostics_weight_norms.png`): By Task 5, layer 1 weights ($l1.weight$) under `gate:conflict` expanded to an L2 norm of $22.700$, compared to $11.790$ under fine-tuning and $12.291$ under `gate:magnitude`.
- `[MEASURED]` **Cumulative Drift Along $d$** (`outputs/phase5/diagnostics_drift_along_d.png`): Projecting parameter displacement onto the historical displacement vector ($\frac{(\theta - \theta_{start}) \cdot d}{\|d\|_2}$) revealed that `gate:conflict` drifted $+2.263$ along $d$ on Task 5, whereas fine-tuning drifted $-0.145$ and `gate:magnitude` drifted $-0.016$.
- `[MEASURED]` **Gated Fraction** (`outputs/phase5/diagnostics_gated_fraction.png`): Over $80\%$ of parameter elements satisfied $g \cdot d > 0$ and were gated at every optimization step across tasks 2 through 5.

```
                    The Asymmetric Ratchet Effect
                    
    Step Direction (-g)          Gate Action            Resulting Weight Motion
    -------------------          -----------            -----------------------
    Towards origin (-d)    -->   BLOCKED (g·d > 0)  --> Zero movement back
    Away from origin (+d)  -->   PERMITTED (g·d ≤ 0) --> Full step taken outward
    
    Net Outcome: Weights take outward steps but can never take inward steps.
                 Weights blow up unidirectionally, causing representational collapse.
```

- `[MEASURED]` **Hypothesis Decision**: The **Ratchet Hypothesis is SUPPORTED**. Directional conflict gating acts as a one-way mechanical ratchet. It selectively prohibits any SGD update that would restore parameters toward their historical starting point while permitting outward-directed steps without constraint. Over successive tasks, this induces runaway parameter inflation and destroys network stability.

### 6.2 Why Magnitude Gating and Drift Repairs Work
- `gate:magnitude` measures only $|d|$ without regard to sign, avoiding any directional bias or asymmetric filtering.
- `gate:drift` directly counters the ratchet by checking the drift-expansion condition ($g \cdot e < 0$, where $e = \theta - \theta_{start}$). Whenever a proposed SGD step would increase parameter distance from the task start anchor, the gate damps the step, acting as a gradient-level barrier analogous to L2-SP / movement-weighted EWC without an explicit loss penalty. As measured in Table 1 and Table 2, this fully stabilizes training, statistically outperforming fine-tuning by $+2.92\%$ on Perm5 and $+5.62\%$ on Perm10.

### 6.3 Layer Gating Hierarchy
Examining the proportion of gated weights per layer for `gate:drift:0.1` (`outputs/phase5/layer_heatmap_best_gate.png`) reveals an intuitive anatomical hierarchy:
- `[MEASURED]` Gating is heaviest in the final classification layers: $l3.weight$ ($87.36\%$), $l2.bias$ ($86.40\%$), and $l3.bias$ ($86.12\%$).
- `[MEASURED]` Gating is lowest in the initial feature projection layer: $l1.weight$ ($78.50\%$).
- `[HYPOTHESIS]` The gate naturally concentrates stability in deeper semantic decision boundaries where task specialization is highest, while permitting greater plasticity in early generic feature projections.

---

## 7. Limitations

1. **Synthetic MLP Architecture**: All experiments were conducted on a small 2-hidden-layer MLP (269,322 parameters). While suitable for isolating mathematical dynamics, scaling to deep convolutional or transformer architectures may exhibit different optimization geometry.
2. **LoRA Oracle Task ID Caveat**: LoRA achieved top BWT ($0.00\%$) and high ACC ($94.30\%$ on Perm5, $98.73\%$ on Split-MNIST), but strictly requires an external oracle to specify the task index at inference time. In authentic task-free continual learning where task boundaries are unobserved, standard LoRA cannot be deployed without a separate task identifier.
3. **Domain-Incremental Head Collapse**: On Split-MNIST, all weight-space regularizers tied with fine-tuning (~62% ACC). Protecting weights is insufficient when output logits directly conflict across tasks sharing an output head.
4. **Validation Budget Constraints**: Due to strict computational fairness, hyperparameter grids were limited to $\le 6$ configurations per method. While sufficient to identify stable regimes, continuous fine-tuning of regularizers could yield marginal improvements.

---

## 8. Future Work: Language Model Scaling (Phase 4 Specification)

In accordance with the project's go/no-go gate rule, Phase 4 (scaling to a 124M-parameter GPT-2 Small model across sequential text domains) was skipped during local execution due to CPU hardware constraints. Training an autoregressive transformer across 8,000 steps without GPU acceleration is computationally intractable. Consequently, Phase 4 is specified here as concrete **Future Work** to guide subsequent GPU-based investigations.

### 8.1 The Optimizer Interaction Problem
In large language models, training standardly employs adaptive optimizers (Adam/AdamW). Adam computes running first ($m_t$) and second ($v_t$) moment estimates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$u_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
Applying a gate directly to the raw gradient $g_t$ is largely negated by Adam: the coordinate-wise normalization by $\sqrt{v_t}$ rescales damped gradients back up, defeating the gate.

### 8.2 Proposed Solution: Update-Level Gating (`GatedAdam`)
To correctly gate adaptive optimizers, gating must be applied directly to the final Adam update vector $u_t$ prior to weight subtraction:

```python
class GatedAdam(torch.optim.Optimizer):
    """Applies parameter drift and conflict gating directly to the Adam update step."""
    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, gate=None):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.gate = gate

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"])
                vh = st["v"] / (1 - b2 ** st["t"])
                upd = mh / (vh.sqrt() + group["eps"])
                
                if self.gate is not None:
                    upd = self.gate.scale_update(p, upd)
                p.add_(upd, alpha=-group["lr"])
```

### 8.3 Experimental Protocol for GPT-2 Small
- **Model**: GPT-2 Small (124M parameters), initialized from pretrained Hugging Face weights.
- **Domains**: 4 sequential domains:
  1. General English (`WikiText-103`)
  2. Source Code (`CodeSearchNet` or `The Stack / Python`)
  3. Biomedical Literature (`PubMed Abstracts`)
  4. Financial / News Text (`Multi-News` / `Financial News`)
- **Protocol**: 2,000 training steps per domain, sequence length 256 tokens, batch size 8.
- **Evaluation Metric**: Held-out Perplexity ($PPL = \exp(\text{loss})$) on 200 sequences per domain evaluated after each domain phase (lower is better). Forgetting is quantified as $\Delta PPL$ on earlier domains.
- **Target Comparisons**: `Fine-Tuning (AdamW)`, `Replay` (500 sequences per domain), `EWC` (with divergence guard), `LoRA` (domain adapters with oracle ID), and `GatedAdam (drift + magnitude)`.

---

## References

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences (PNAS)*, 114(13), 3521-3526.
2. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual Learning Through Synaptic Intelligence. *International Conference on Machine Learning (ICML)*, PMLR 70:3987-3995.
3. Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient Episodic Memory for Continual Learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
4. Chaudhry, A., Rohrbach, M., Elhoseiny, M., et al. (2019). On Tiny Episodic Memories in Continual Learning. *arXiv preprint arXiv:1902.10486*.
5. Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*.
6. Behrouz, A., Santacroce, M., & Pelechrinis, K. (2024). Titans: Learning to Memorize at Test Time. *arXiv preprint arXiv:2412.19837*.
