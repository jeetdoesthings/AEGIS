# Poster Presentation: Continual Learning with Directional & Magnitude Gradient Gating

**Title:** Beyond the Ratchet: Empirical Evaluation, Mechanistic Failure Modes, and Anti-Ratchet Repairs for Gradient-Conflict Gating in Continual Learning  
**Author:** Project AEGIS  
**Department:** Computer Science & Machine Learning  

---

## 1. The Problem: Catastrophic Forgetting

- **Catastrophic Forgetting:** When neural networks are trained on new distributions sequentially, gradient descent aggressively overwrites weights critical for earlier tasks.
- **Why Not RAG or Long-Context Windows?**
  - *RAG* only retrieves external facts into prompts; it leaves weights unchanged and cannot acquire procedural skills or non-textual perceptual capabilities.
  - *Long context* is ephemeral, memory-heavy, and vanishes once an inference session closes.
  - Lifelong learning requires consolidating knowledge directly into persistent network weights without unbounded data storage.

---

## 2. Experimental Setup & Fair Benchmark Protocol

- **Benchmarks:**
  1. *Permuted-MNIST (5 tasks):* Spatial pixel permutations of digit images (784 inputs, 10 classes).
  2. *Permuted-MNIST (10 tasks):* 10 sequential permutations testing long-horizon compounding degradation.
  3. *Split-MNIST (5 tasks, domain-incremental):* Disjoint digit pairs (0/1, 2/3, 4/5, 6/7, 8/9) sharing a single 2-way output head.
- **Model:** 2-hidden-layer MLP ($784 \to 256 \to 256 \to C$, 269k parameters), plain SGD ($\eta = 0.01$).
- **Fairness Protocol:**
  - Standard MNIST split into 55,000 train, 5,000 validation (held-out tuning), and 10,000 test.
  - Equal tuning budget ($\le 6$ configs per method on validation seeds 0–2).
  - Best configs locked before 5 final evaluation seeds (100–104) on held-out test split.
  - **Statistical Tie Rule:** Overlapping `mean ± std` intervals are reported as ties, never wins.

---

## 3. Results Summary Table

*Evaluated on held-out test splits across 5 independent random seeds (100–104). Values shown as mean ± std (%).*

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

*\*LoRA requires an oracle task ID at test time to select task-specific adapters.*

---

## 4. Main Findings

1. **Original Directional Gate Collapses via an Asymmetric Ratchet:**
   - Naively zeroing updates where $g \cdot d > 0$ blocks steps opposing historical displacement but lets outward steps proceed unhindered.
   - Result: Weights blow up unidirectionally ($l1.weight$ norm doubles to 22.70 vs 11.79 for fine-tuning), and Perm10 accuracy collapses to $41.92\%$ ($-36.29\%$ below fine-tuning).
2. **Magnitude Gating Succeeds Without Direction:**
   - The magnitude ablation completely ignores direction and damps weights based purely on $|d|$.
   - Result: Outperforms fine-tuning on Perm10 ($83.13\%$ vs $78.21\%$) and cuts backward forgetting from $-20.80\%$ to $-12.82\%$.
3. **Anti-Ratchet Repairs Statistically Outperform Fine-Tuning:**
   - Adding a quadratic distance penalty (`gate:drift`) or hybrid damping (`gate:hybrid`) halts weight runaway.
   - On Perm10, `gate:hybrid` achieves $83.82 \pm 0.54\%$ and `gate:drift` achieves $83.21 \pm 1.26\%$ ACC, statistically outperforming plain fine-tuning ($78.21 \pm 1.68\%$) by $>+5\%$ without memory buffers.
4. **The Output Head Bottleneck:**
   - On domain-incremental Split-MNIST, all buffer-free weight-space methods tie around $62\%$ ACC. Protecting hidden representations fails when output logits overwrite each other. Replay ($96.85\%$) or task routing ($98.73\%$) is required.

---

## 5. Limitations & Future Work

- **Limitations:** Evaluated on a 2-layer MLP on permutation and split benchmarks; LoRA's zero forgetting relies on test-time task IDs; weight gating alone cannot resolve shared output head interference without exemplars.
- **Next Step (Phase 4):** Scaling to language models—fine-tuning GPT-2 Small (124M) sequentially across 4 text domains (WikiText-103, Code, PubMed, News) using **`GatedAdam`** to prevent adaptive moment normalization from cancelling gradient gates.
