# Project AEGIS: Executive Overview & Simplified Project Brief
## Continual Learning, Catastrophic Forgetting, and Parameter-Space Gradient Gating

**Author:** Jeet & Project AEGIS Team  
**Institution:** Machine Learning Research Lab  
**Repository:** [github.com/jeetdoesthings/AEGIS](https://github.com/jeetdoesthings/AEGIS)  
**Standalone PDF Brief:** [`outputs/final/AEGIS_Simplified_Project_Overview.pdf`](outputs/final/AEGIS_Simplified_Project_Overview.pdf) (3 pages, executive layout)  
**Comprehensive Technical Report:** [`outputs/final/AEGIS_Master_Project_Report.pdf`](outputs/final/AEGIS_Master_Project_Report.pdf) (6 pages, exhaustive monograph)  
**Date:** November 2026  

---

## 📌 Project Status at a Glance: 100% Core Roadmap Complete

| Phase & Milestone | Completion | Key Delivered Outcomes | Verification Status |
| :--- | :---: | :--- | :---: |
| **Phase 0: Baseline & Freeze** | **100%** | `cl.py` frozen permanently; baseline timings verified on Apple Silicon M4 CPU. | Passed (`19df00d`) |
| **Phase 1: Baselines & EWC Fix** | **100%** | 55k/5k/10k splits; EWC NaN explosion solved; $\lambda=20$ locked; Perm5 test run. | Passed (`2ec680e`, `b1ca604`) |
| **Phase 2: Harder Benchmarks** | **100%** | 10-task Permuted-MNIST; Split-MNIST domain-incremental run; 5 test seeds evaluated. | Passed (`f5a03e6`, `9eac5c3`) |
| **Phase 3: Diagnostics & Repairs** | **100%** | Ratchet effect diagnosed; `gate:drift` & `gate:hybrid` statistically beat FT by $>+5\%$. | Passed (`00453dc`) |
| **Phase 4: Language Model Scaling** | **Scoped** | Skipped on local CPU per go/no-go rule; `GatedAdam` designed for future GPU work. | Documented as Future Work |
| **Phase 5: Deliverables Pack** | **100%** | Tables (CSV/PNG); forgetting curves; heatmaps; research report; claim audit; Q&A. | Passed (`572c750`, `7e9076c`) |

---

## 1. What is this project about?

### The Core Problem (Catastrophic Forgetting)
When artificial neural networks are trained on task A, and then sequentially trained on task B, the gradient descent updates required to master task B overwrite the weights critical for task A. Within a few updates, the network completely forgets how to perform task A. This fundamental limitation is called **catastrophic forgetting**. It represents the single greatest barrier to building autonomous, lifelong learning AI agents.

### Why RAG and Long Context Windows Don't Fix This
In modern AI workflows, two popular paradigms are frequently assumed to bypass forgetting:
1. **Retrieval-Augmented Generation (RAG):** RAG injects external text into an inference prompt. However, it leaves the actual neural network weights static; it cannot acquire new procedural reasoning skills, update visual perceptual representations, or adapt neural policies over time.
2. **Long-Context Windows:** In-context learning across large prompt buffers is temporary and memory-heavy. It incurs high computational costs, applies only during an active session, and vanishes completely when inference closes.

True AI autonomy requires lifelong learning directly inside the network's persistent parameter weights without storing unbounded raw data.

### The Research Question
Can we create a **buffer-free** mathematical gate directly on gradients that shields critical weights without storing old data (no memory replay buffers)? 

We investigated the **Gradient-Conflict Gate**—an algorithm that tracks historical parameter movement across prior tasks ($d = \theta_{start} - \theta^{(0)}$) and attempts to damp incoming gradient updates ($g$) that oppose historical trajectories.

---

## 2. What have we done in this project?

The project was conducted in five sequential, rigorously documented stages:

1. **Diagnosed & Fixed Broken Baselines (Phase 0 vs Phase 1):** The starting codebase (`cl.py`) had an EWC baseline that mysteriously scored 9.80% accuracy. We uncovered the mathematical cause: default penalty $\lambda=5000$ with learning rate $0.01$ caused exponential gradient explosion by step 13 into NaNs. In PyTorch, `torch.argmax` on NaNs defaulted to digit '0' (which happens to be exactly 9.8% of MNIST!). We added divergence guards, swept $\lambda$, locked optimal $\lambda=20$ (achieving 93.76% ACC), froze `cl.py` permanently, and rebuilt the engine cleanly in `cl2.py`.
2. **Established Honest, Leakage-Free Evaluation:** Partitioned MNIST into 55,000 train, 5,000 validation (held-out tuning), and 10,000 test images. Tuned all 7 methods across 25 configurations with equal tuning budgets ($\le 6$ configs per method) strictly on the validation split.
3. **Uncovered the 'Asymmetric Ratchet' Failure:** On 10 tasks, the original directional conflict gate (`gate:conflict`) collapsed to **41.92% ACC** ($-36.29\%$ below plain fine-tuning!). Using step-level telemetry (`diagnostics.py`), we discovered why: blocking opposing updates while letting outward updates run unhindered acted as a mechanical ratchet—layer 1 weight norms doubled from 11.79 to 22.70 and ran away along $d$ (+2.263 vs -0.145).
4. **Invented Stabilized Anti-Drift Repairs:** Formulated an anti-drift gate (`gate:drift`, which brakes steps that expand distance from starting weights $e = \theta - \theta_{start}$) and a hybrid gate (`gate:hybrid`). These repairs eliminated parameter runaway: on 10 tasks, `gate:hybrid` (83.82%) and `gate:drift` (83.21%) **statistically beat fine-tuning (78.21%) by >+5% ACC** without storing replay data.
5. **Tested Multi-Benchmark Boundaries:** Discovered that on domain-incremental Split-MNIST, all buffer-free methods tie at ~62% ACC, revealing that weight-space gating cannot resolve shared output-head interference without data replay ($96.85\%$) or task routing (LoRA: $98.73\%$).

---

## 3. What were the results?

*All evaluations conducted on held-out test splits across 5 independent random seeds (100–104). Values reported as **mean ± std (%)**. Data source: [`outputs/phase5/master_summary.csv`](outputs/phase5/master_summary.csv).*

| Method | Category | Perm5 ACC (%) | Perm10 ACC (%) | Split ACC (%) | Key Finding / Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Fine-Tuning (Plain SGD)** | Baseline | $88.97 \pm 1.85$ | $78.21 \pm 1.68$ | $62.07 \pm 1.00$ | Reference baseline. Forgetting doubles on 10 tasks ($-20.8\%$ BWT). |
| **EWC (repaired, $\lambda=20/100$)** | Regularizer | $93.76 \pm 0.36$ | $90.22 \pm 0.87$ | $63.96 \pm 1.39$ | Strong on permutations ($+12.0\%$ on Perm10); ties on Split-MNIST. |
| **Replay (buffer=1000)** | Exemplar Buffer | $\mathbf{95.85 \pm 0.11}$ | $\mathbf{95.22 \pm 0.09}$ | $96.85 \pm 0.30$ | Top overall accuracy on all benchmarks; stores 1,000 raw samples. |
| **LoRA ($r=16/8$, oracle ID)\*** | Architecture | $94.30 \pm 0.12$ | $94.05 \pm 0.05$ | $\mathbf{98.73 \pm 0.06}$ | Zero backward forgetting ($0.0\%$ BWT); requires test-time task ID. |
| **Gate: Conflict (original, $\beta=0.25$)** | Original Gate | $81.73 \pm 1.41$ | $41.92 \pm 1.71$ | $62.28 \pm 0.98$ | Catastrophic failure ($-36.3\%$ on Perm10); ratchet runaway. |
| **Gate: Magnitude (ablation, $\beta=1.0$)** | Ablation | $91.43 \pm 0.86$ | $83.13 \pm 1.22$ | $61.90 \pm 0.36$ | Beats conflict gate by $+41\%$ on Perm10; magnitude alone works. |
| **Gate: Drift (anti-drift, $\beta=0.1$)** | Repaired Gate | $91.89 \pm 1.03$ | $83.21 \pm 1.26$ | $62.62 \pm 1.03$ | Statistically beats fine-tuning by about 5 points ($+5.0\%$) on Perm10 without buffers. |
| **Gate: Hybrid (conflict×mag, $\beta=0.1$)** | Repaired Gate | $90.98 \pm 0.90$ | $83.82 \pm 0.54$ | $62.78 \pm 0.56$ | Statistically beats fine-tuning by about 5 points ($+5.6\%$) on Perm10 without buffers. |

*\*LoRA requires an oracle task ID at test time to route representations to task-specific adapters.*

> [!IMPORTANT]
> **Honest Comparative Framing & Performance Boundaries:**
> - **Repaired EWC Outperforms Every Gate:** Repaired EWC (Perm10 ACC $90.22 \pm 0.87\%$) outperforms every gate variant ($83.13–83.82\%$) by over 6.4 percentage points. The gate does not compete with EWC on retention.
> - **Gate Variants Beat Fine-Tuning by ~5 Points:** On 10 tasks, `gate:hybrid` ($83.82\%$) and `gate:drift` ($83.21\%$) statistically outperform plain fine-tuning ($78.21\%$) by about 5 points without memory buffers.
> - **Split-MNIST Buffer-Free Ceiling (62–64%):** On domain-incremental Split-MNIST, no method without replay ($96.85\%$) or task routing (LoRA: $98.73\%$) beats $62–64\%$ ACC. Fine-tuning ($62.07\%$), EWC ($63.96\%$), and all gates ($61.90–62.78\%$) form an inseparable statistical tie due to shared-head binary logit overwrite.

---

## 4. What are the outputs?

All deliverables are organized in `outputs/` and version-controlled:

1. **Master Documents & Executive Briefs:**
   - [`outputs/final/AEGIS_Simplified_Project_Overview.pdf`](outputs/final/AEGIS_Simplified_Project_Overview.pdf): 3-page executive summary PDF.
   - [`outputs/final/AEGIS_Master_Project_Report.pdf`](outputs/final/AEGIS_Master_Project_Report.pdf): 6-page comprehensive technical monograph with embedded figures.
   - [`outputs/phase5/report.md`](outputs/phase5/report.md): Formal 8-section academic research paper draft.
   - [`outputs/phase5/claim_audit.md`](outputs/phase5/claim_audit.md): Epistemological audit verifying all claims against raw numbers under the tie rule.
2. **Defense & Presentation Suite:**
   - [`outputs/phase5/poster_text.md`](outputs/phase5/poster_text.md): Publication-ready poster layout text.
   - [`outputs/phase5/qa.md`](outputs/phase5/qa.md): 15 examiner defense questions with exact empirical numbers.
   - [`outputs/phase5/script_60s.md`](outputs/phase5/script_60s.md): 60-second spoken poster script (~140 words).
3. **Data Tables & Visual Charts:**
   - [`outputs/phase5/master_summary.csv`](outputs/phase5/master_summary.csv) / [`master_summary.png`](outputs/phase5/master_summary.png): 10-method cross-benchmark master comparison.
   - [`outputs/phase5/forgetting_curve_perm10_all.png`](outputs/phase5/forgetting_curve_perm10_all.png): 10-task retention curve plot.
   - [`outputs/phase5/diagnostics_weight_norms.png`](outputs/phase5/diagnostics_weight_norms.png) & [`diagnostics_drift_along_d.png`](outputs/phase5/diagnostics_drift_along_d.png): Parameter telemetry confirming the ratchet mechanism.
   - [`outputs/phase5/layer_heatmap_best_gate.png`](outputs/phase5/layer_heatmap_best_gate.png): Layer gating hierarchy heatmap.

---

## 5. What is the completion percentage?

- **Core Empirical & Deliverable Roadmap: 100% Complete.** Every planned experiment across Permuted-MNIST (5 and 10 tasks), domain-incremental Split-MNIST, validation tuning sweeps, EWC diagnosis, parameter telemetry, and gate repairs has been executed across 5 test seeds, audited, and committed.
- **Phase 4 (Language Models): Scoped as Future Work.** Phase 4 was deliberately skipped during local execution per the plan's go/no-go rule because execution was on local CPU hardware (where training a 124M-parameter transformer across 8,000 steps without GPU acceleration is computationally intractable). The theoretical blueprint, the Adam optimizer moment interaction, and the `GatedAdam` algorithm are fully specified in Section 6 of the Master Report for immediate GPU deployment.

---

## 6. How to Run & Verify

```bash
# 1. Fast smoke test (10 seconds)
python cl2.py --seeds 1 --epochs 1

# 2. Evaluate repaired gates across 5 seeds on all benchmarks
python tune_and_run_repaired.py

# 3. Regenerate step-by-step diagnostic telemetry
python diagnostics.py

# 4. Regenerate master tables and PDFs
python build_results_pack.py
python build_master_pdf.py
python build_simplified_pdf.py
```
