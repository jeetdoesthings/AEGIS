# Viva & Examiner Q&A Guide: Continual Learning Experiment

This document details the 15 most likely examiner questions for project defense, accompanied by concise answers grounded in raw measured data from `outputs/phase5/`. Answers explicitly flag instances where the empirical results do not support a strong claim.

---

### Q1: What datasets and benchmarks did you use, and where do they come from?
**Answer:** We evaluated across three benchmarks derived from torchvision's standard MNIST handwritten digit dataset (60,000 train, 10,000 test):
1. **Permuted-MNIST 5-Task**: 5 sequential tasks where each task applies a distinct fixed pseudo-random spatial permutation across all 784 pixels.
2. **Permuted-MNIST 10-Task**: 10 sequential permutations testing compounding degradation over a longer horizon.
3. **Split-MNIST 5-Task**: Domain-incremental classification of disjoint digit pairs (0/1, 2/3, 4/5, 6/7, 8/9) with a shared 2-way output head.

---

### Q2: Why did you use a small 2-layer MLP instead of a larger modern architecture?
**Answer:** A small MLP ($784 \to 256 \to 256 \to C$, 269k parameters) allows isolating exact weight-space and optimization dynamics under plain SGD without confounding factors such as attention masking, layer normalization, residual stream routing, or learning-rate warmups. Additionally, on our M4 local machine, CPU forward-backward passes had lower dispatch latency (0.156s vs 0.317s for 500 passes on MPS), enabling exhaustive multi-seed tuning sweeps across hundreds of training epochs.

---

### Q3: How exactly is catastrophic forgetting quantified in your experiments?
**Answer:** We evaluate test accuracy on all tasks after learning each task, constructing the matrix $R \in \mathbb{R}^{T \times T}$ where $R_{i, j}$ is accuracy on task $j$ after training on task $i$. Forgetting is quantified by Backward Transfer ($BWT$):
$$BWT = \frac{1}{T-1} \sum_{j=1}^{T-1} (R_{T, j} - R_{j, j})$$
A negative $BWT$ indicates forgetting. In plain fine-tuning, $BWT$ drops from $-9.62 \pm 2.34\%$ on 5 tasks to $-20.80 \pm 1.84\%$ on 10 tasks, and plunges to $-46.36 \pm 1.23\%$ on Split-MNIST.

---

### Q4: Why did you split the MNIST training set into train and validation sets instead of tuning directly on the test set?
**Answer:** Tuning hyperparameters directly on the test set is methodological data leakage that inflates reported performance and yields brittle configurations. We partitioned MNIST's 60,000 training images into 55,000 training samples and 5,000 validation samples (fixed seed 42). All tuning sweeps (seeds 0–2, $\le 6$ configs per method) were selected strictly on validation ACC, locked into JSON configurations, and evaluated once on the untouched 10,000 test images across fresh seeds 100–104.

---

### Q5: What does your negative result on the original gradient-conflict gate mean?
**Answer:** `[MEASURED]` The original directional conflict gate (`gate:conflict`, $\beta=0.25$) failed severely: it underperformed plain fine-tuning by $-7.24 \pm 1.38\%$ on Perm5 ($81.73\%$ vs $88.97\%$) and collapsed on Perm10 to $41.92 \pm 1.71\%$ ($-36.29 \pm 2.82\%$ vs fine-tuning's $78.21\%$).  
`[HYPOTHESIS]` Rather than an implementation flaw, this represents an inherent mathematical failure mode: directional gating creates an *asymmetric ratchet*. Blocking updates where $g \cdot d > 0$ prevents steps back toward origin while allowing outward steps ($g \cdot d \le 0$) without constraint, causing parameter norms to double ($l1.weight$ norm 22.70 vs 11.79) and drift away uncontrollably.

---

### Q6: Why did EWC originally fail with 9.8% accuracy, and how did you repair it?
**Answer:** The initial codebase set EWC penalty $\lambda = 5000$. With SGD learning rate $\eta = 0.01$ and unnormalized Fisher diagonal values, the update term $\eta \lambda F_i \gg 1$. At step 13 of Task 2, gradient updates diverged exponentially into NaNs and Infinities. `torch.argmax` on NaN logits defaulted to class 0, producing exactly 9.80% accuracy (the class frequency of digit 0). We diagnosed this (`outputs/phase1/ewc_diagnosis.txt`), added an explicit divergence guard, and swept $\lambda \in \{1, 5, 20, 100\}$. On validation, $\lambda=20$ achieved optimal stability ($93.66 \pm 0.73\%$), and on test, repaired EWC achieved $93.76 \pm 0.36\%$ on Perm5 and $90.22 \pm 0.87\%$ on Perm10 without diverging.

---

### Q7: LoRA achieved 0.00% forgetting. Why isn't LoRA considered the outright winner?
**Answer:** LoRA achieved $BWT = 0.00 \pm 0.00\%$ and high ACC ($94.30\%$ on Perm5, $94.05\%$ on Perm10, $98.73\%$ on Split-MNIST) because it isolates task updates into separate low-rank adapter matrices ($W = W_0 + B_t A_t$). However, **LoRA requires an oracle task ID at test time** to select which adapter to route the input through. In genuine task-free continual learning where task IDs are unknown at inference, standard LoRA cannot function without an external task classifier.

---

### Q8: What is actually new or novel in your work?
**Answer:** We explicitly state that the gradient-conflict gate is **not novel**. Tracking parameter displacement $d = \theta - \theta_{start}$ relates directly to Synaptic Intelligence (Zenke et al., 2017), and directional gradient conflict projection derives from GEM (Lopez-Paz & Ranzato, 2017). What is novel in our work is the **rigorous mechanistic autopsy**: uncovering the asymmetric ratchet mechanism via step-level telemetry, empirically demonstrating that magnitude gating beats directional conflict, and designing anti-ratchet repairs (`gate:drift` and `gate:hybrid`) that turn a failing method into one that statistically beats fine-tuning by $>+5\%$ on Perm10.

---

### Q9: Did your magnitude gate beat the conflict gate? What does that indicate?
**Answer:** `[MEASURED]` Yes, decisively. On Perm5, `gate:magnitude` achieved $91.43 \pm 0.86\%$ ACC vs $81.73 \pm 1.41\%$ for `gate:conflict` ($+9.70\%$ margin). On Perm10, `gate:magnitude` scored $83.13 \pm 1.22\%$ vs $41.92 \pm 1.71\%$ for `gate:conflict` ($+41.21\%$ margin).  
`[HYPOTHESIS]` This indicates that checking the *sign* of $g \cdot d$ is actively harmful because it induces directional asymmetry. Dampening updates solely based on the *magnitude* of historical movement ($|d|$) provides effective regularization without the runaway ratchet.

---

### Q10: How does gate:drift work in your code, and how do your repaired gates compare to fine-tuning?
**Answer:**
- **Mechanism in Code (`cl2.py`)**: `gate:drift` is an anti-drift gradient gate, not an additive loss penalty. At each step, it computes $e = \theta - \theta_{start}$ and checks $\text{cond} = (g \cdot d > 0) \lor (g \cdot e < 0)$. If an update moves parameters further from the task starting anchor ($g \cdot e < 0$) or opposes historical displacement ($g \cdot d > 0$), it damps the gradient by $1 - \beta \cdot \text{mag} \cdot \text{cond}$. In the continual learning literature, penalizing movement away from starting weights is standardly done via additive quadratic loss penalties (L2-SP or EWC). Here, `gate:drift` enforces that anchor proximity directly in the gradient domain without an explicit loss penalty.
- **Empirical Results vs Fine-Tuning**:
  - On Perm5: `gate:drift` achieved $91.89 \pm 1.03\%$ ACC vs $88.97 \pm 1.85\%$ for fine-tuning. The intervals $[90.86, 92.92]$ and $[87.12, 90.82]$ do not overlap, so **`gate:drift` statistically outperforms fine-tuning by $+2.92 \pm 0.96\%$**.
  - On Perm10: `gate:hybrid` achieved $83.82 \pm 0.54\%$ and `gate:drift` achieved $83.21 \pm 1.26\%$ ACC vs $78.21 \pm 1.68\%$ for fine-tuning. Both intervals ($[83.28, 84.36]$ and $[81.95, 84.47]$) are completely disjoint from fine-tuning's $[76.53, 79.89]$, **statistically outperforming fine-tuning by $>+5\%$ ACC**.

---

### Q11: How do your repaired gates compare to Experience Replay?
**Answer:** ⚠️ **Flagged: Results do NOT support claiming the gate beats replay.**  
`[MEASURED]` Experience Replay (with 1,000 stored exemplars) outperformed all buffer-free gating methods on all three benchmarks: $95.85 \pm 0.11\%$ on Perm5, $95.22 \pm 0.09\%$ on Perm10, and $96.85 \pm 0.30\%$ on Split-MNIST. While `gate:drift` ($83.21\%$) and `gate:hybrid` ($83.82\%$) beat fine-tuning on Perm10, they trail replay by ~11.5% ACC. Replay directly re-anchors the loss surface using real historical data, which no buffer-free weight-space method in our study matched.

---

### Q12: Why did all buffer-free methods fail on Split-MNIST (~62% ACC)?
**Answer:** ⚠️ **Flagged: Results do NOT support claiming any gate succeeds on Split-MNIST.**  
`[MEASURED]` On Split-MNIST, fine-tuning, EWC, and all six gate variants collapsed into a mutual statistical tie between $61.90\%$ and $63.96\%$ ACC, suffering backward transfer between $-43.76\%$ and $-46.39\%$.  
`[HYPOTHESIS]` Split-MNIST is domain-incremental with a shared 2-way output head. Learning task 2 (digits 2 and 3) directly overwrites the output weights assigned to binary logits 0 and 1. Protecting hidden representations cannot resolve representational collision at the output head without exemplar replay ($96.85\%$) or task routing (LoRA, $98.73\%$).

---

### Q13: What did you discover about layer-wise gating hierarchy?
**Answer:** `[MEASURED]` Telemetry of `gate:drift:0.1` (`outputs/phase5/layer_heatmap_best_gate.png`) showed that gating frequency varies systematically across depth:
- Classification head: $l3.weight$ (87.36%), $l3.bias$ (86.12%), $l2.bias$ (86.40%).
- Feature extractor: $l1.weight$ (78.50%).  
`[HYPOTHESIS]` The gate automatically preserves high-level task decision boundaries more aggressively while leaving early feature layers relatively plastic.

---

### Q14: Why is Forward Transfer (FWT) close to zero across all methods?
**Answer:** `[MEASURED]` $FWT$ measures accuracy on task $j$ prior to training on it, compared to a randomly initialized network ($FWT = \frac{1}{T-1} \sum_{j=2}^T (R_{j-1, j} - R_{0, j})$). Across all methods on Permuted-MNIST, $FWT$ remained between $+0.5\%$ and $+2.0\%$.  
`[HYPOTHESIS]` This is expected: each task applies an independent random pixel permutation that completely destroys spatial correlations. Learning permutation $A$ provides zero inductive prior or feature reusability for permutation $B$.

---

### Q15: What is the status of Phase 4, and how would you scale this to modern Language Models?
**Answer:** Phase 4 (evaluating a 124M-parameter GPT-2 Small across 4 sequential text domains) was **skipped per the plan's go/no-go decision rule**, which mandates skipping Phase 4 when running on local CPU without dedicated GPU compute. Running a 124M transformer on CPU across 8,000 optimization steps is computationally intractable. Consequently, Phase 4 is presented strictly as **Future Work**.  
For that future implementation, standard gradient gating would fail under Adam because coordinate-wise second-moment normalization ($\sqrt{v_t}$) rescales damped gradients back up. We designed **`GatedAdam`**, which applies gating directly to the final Adam update vector ($u_t = \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$) before parameter subtraction ($p \leftarrow p - \eta u_{gated}$). This preserves the optimizer's moment tracking while enforcing drift protection.
