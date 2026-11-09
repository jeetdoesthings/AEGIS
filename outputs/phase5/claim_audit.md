# Claim Audit: Text Verification & Resolution Log

This document audits all occurrences of sensitive epistemological and comparative terms—**"prove"**, **"solve"**, **"solved"**, **"beats"**, **"outperforms"**, **"novel"**, and **"state-of-the-art"** (and their grammatical derivatives)—across the project deliverables in `outputs/phase5/` and related analytical texts.

Every detected instance has been inspected against the raw experimental data files and evaluated under the **Statistical Tie Rule**:
> *Two methods are reported as a statistical tie if their `mean ± std` intervals overlap. Claims of 'beating' or 'outperforming' are prohibited for overlapping intervals and are permitted only when empirical distributions are strictly disjoint.*

---

## 1. Audit Summary Table

| Term / Issue Checked | Occurrences Found | Compliant / Supported | Rewritten / Resolved |
| :--- | :---: | :---: | :---: |
| **novel / novelty** | 2 | 2 (Both explicitly disclaim novelty) | 0 |
| **state-of-the-art / sota** | 0 | 0 | 0 |
| **prove / proved / proves** | 4 | 0 (Banned from empirical claims) | 4 (Resolved to "supports" / "indicates" / "suggests") |
| **solve / solved / solves** | 1 | 1 (Contextual usage: task objective) | 0 |
| **beats** | 1 | 1 (Rule definition text only) | 0 |
| **outperform / outperforming** | 6 | 5 (Supported by non-overlapping intervals) | 1 (Rewritten to note statistical tie) |
| **Split-MNIST Explanation Label** | 2 | 1 (Correctly labeled in report.md) | 1 (Relabeled from MEASURED to HYPOTHESIS in summary texts) |
| **gate:drift Mathematical Formulation** | 2 | 0 (Incorrectly described as loss penalty) | 2 (Corrected to exact gradient gate condition from `cl2.py`) |
| **Phase 4 Milestone Status** | 2 | 0 (Incorrectly implied completed) | 2 (Clarified as skipped per go/no-go rule; Future Work) |

---

## 2. Item-by-Item Verification & Resolution

### Hit 1: "novel" / "novelty"
- **Location**: [`outputs/phase5/report.md:48`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L48)
- **Text**: *"The Gradient-Conflict Gate investigated here is **not novel**. It is conceptually derived from the intersection of Synaptic Intelligence (tracking cumulative weight displacement $d = \theta - \theta_{start}$) and GEM (penalizing or zeroing gradient updates that exhibit directional conflict). Rather than claiming algorithmic novelty, this work systematically uncovers the exact mathematical failure modes of naive directional gradient gating, diagnoses its mechanistic collapse, and formulates stable, mathematically grounded repairs."*
- **Audit Assessment**: **COMPLIANT**. The text explicitly and unambiguously disclaims novelty, correctly attributing lineage to Synaptic Intelligence and Gradient Episodic Memory.

---

### Hit 2: "solve"
- **Location**: [`outputs/results_explanation.md:59`](file:///Users/jeet/Desktop/AEGIS/outputs/results_explanation.md#L59)
- **Text**: *"...continually overwrites parameter coordinates required to solve earlier permutations."*
- **Audit Assessment**: **COMPLIANT**. Used in the standard colloquial sense of classifying a permutation task, not claiming to have "solved catastrophic forgetting".

---

### Hit 3: "beats" / "outperforming"
- **Location**: [`outputs/phase5/report.md:119`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L119)
- **Text**: *"In accordance with rigorous statistical practices, two methods are reported as a **statistical tie** if their `mean ± std` intervals overlap. Claims of 'beating' or 'outperforming' are strictly reserved for non-overlapping distributions."*
- **Audit Assessment**: **COMPLIANT**. This defines the foundational methodological constraint governing all assertions in the paper.

---

### Hit 4: "outperforming" (`gate:magnitude` on Perm5)
- **Location**: [`outputs/phase5/report.md:147`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L147)
- **Original Draft**: *"`gate:magnitude` achieves $91.43 \pm 0.86\%$ ACC, outperforming fine-tuning by $+2.46 \pm 1.73\%$ on paired seeds..."*
- **Data Check**:
  - `gate:magnitude`: $91.43 \pm 0.86\% \implies [90.57\%, 92.29\%]$
  - `finetune`: $88.97 \pm 1.85\% \implies [87.12\%, 90.82\%]$
  - Interval overlap: $[90.57\%, 90.82\%]$ overlaps by $0.25\%$.
- **Audit Assessment**: **VIOLATION OF INDEPENDENT TIE RULE**. Even though the per-seed paired difference is strictly positive ($+2.46 \pm 1.73\%$), their independent mean ± std intervals overlap.
- **Resolution**: **REWRITTEN** to:
  *"`[MEASURED]` The magnitude ablation (`gate:magnitude`) achieves $91.43 \pm 0.86\%$ ACC (paired difference $+2.46 \pm 1.73\%$ over fine-tuning; independent intervals $[90.57, 92.29]$ and $[87.12, 90.82]$ overlap slightly, constituting a statistical tie under the independent interval rule) while cutting backward forgetting in half ($-4.39 \pm 1.01\%$ vs $-9.62 \pm 2.34\%$)."*

---

### Hit 5: "outperforms" (`gate:drift` on Perm5)
- **Location**: [`outputs/phase5/report.md:148`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L148)
- **Text**: *"`[MEASURED]` The repaired `gate:drift` variant achieves $91.89 \pm 1.03\%$ ACC, which statistically outperforms plain fine-tuning ($88.97 \pm 1.85\%$) under the non-overlapping tie rule ($91.89 - 1.03 = 90.86 > 88.97 + 1.85 = 90.82$)."*
- **Data Check**:
  - `gate:drift`: $91.89 \pm 1.03\% \implies [90.86\%, 92.92\%]$
  - `finetune`: $88.97 \pm 1.85\% \implies [87.12\%, 90.82\%]$
  - Lower bound of `gate:drift` ($90.86\%$) strictly exceeds upper bound of `finetune` ($90.82\%$).
- **Audit Assessment**: **VERIFIED & SUPPORTED**. Intervals are non-overlapping. Claim is statistically justified.

---

### Hit 6: "outperform" (`gate:hybrid` and `gate:drift` on Perm10)
- **Location**: [`outputs/phase5/report.md:172`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L172), [`outputs/phase5/captions.md:15`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/captions.md#L15), [`outputs/phase5/report.md:17`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L17)
- **Text**: *"`[MEASURED]` Both repaired variants statistically outperform fine-tuning with non-overlapping distributions: `gate:hybrid` achieves $83.82 \pm 0.54\%$ ACC ($+5.62 \pm 1.21\%$) and `gate:drift` achieves $83.21 \pm 1.26\%$ ACC ($+5.01 \pm 1.48\%$)."*
- **Data Check**:
  - `finetune`: $78.21 \pm 1.68\% \implies [76.53\%, 79.89\%]$
  - `gate:drift`: $83.21 \pm 1.26\% \implies [81.95\%, 84.47\%]$ (Lower bound $81.95\% > 79.89\%$, separation $+2.06\%$).
  - `gate:hybrid`: $83.82 \pm 0.54\% \implies [83.28\%, 84.36\%]$ (Lower bound $83.28\% > 79.89\%$, separation $+3.39\%$).
- **Audit Assessment**: **VERIFIED & SUPPORTED**. Both intervals are completely disjoint from fine-tuning by substantial margins.

---

### Hit 7: "Task 1 Retention on Perm10"
- **Location**: [`outputs/phase5/captions.md:39`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/captions.md#L39)
- **Original Draft**: *"...while repaired gate:hybrid and gate:drift preserve >81% Task 1 accuracy, outperforming fine-tuning by >12% on Task 1 retention."*
- **Data Check**:
  - Task 1 accuracy after Task 10 from `outputs/phase2/final_perm10.json` & `outputs/phase3/final_repaired_perm10.json`:
    - `finetune`: $47.31 \pm 4.80\%$
    - `gate:conflict`: $13.44 \pm 4.44\%$
    - `gate:magnitude`: $65.20 \pm 7.04\%$
    - `gate:drift`: $54.44 \pm 5.51\%$
    - `gate:hybrid`: $52.71 \pm 4.93\%$
  - The $>81\%$ figure was mistakenly carried over from Perm5 ($80.72 \pm 4.13\%$).
- **Audit Assessment**: **CORRECTED**. The caption has been revised to cite the exact measured percentages:
  *"...while gate:magnitude preserves $65.20 \pm 7.04\%$, gate:drift preserves $54.44 \pm 5.51\%$, and gate:hybrid preserves $52.71 \pm 4.93\%$ Task 1 accuracy, compared to $47.31 \pm 4.80\%$ for plain fine-tuning."*

---

### Hit 8: "proved" in Presentation Script
- **Location**: [`outputs/phase5/script_60s.md:16`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/script_60s.md#L16)
- **Original Draft**: *"Using parameter telemetry, we proved why: blocking opposing updates creates an asymmetric ratchet..."*
- **Audit Assessment**: **VIOLATION OF BANNED TERM RULE**. The plan explicitly bans "prove" for empirical observations. Logging step patterns and observing that drift repair helps provides strong supporting evidence, but does not constitute formal mathematical proof.
- **Resolution**: **REWRITTEN** to:
  *"Parameter telemetry supports this mechanism: blocking opposing updates creates an asymmetric ratchet that doubles weight norms and causes runaway drift."*

---

### Hit 9: "prove" in Viva Q&A
- **Location**: [`outputs/phase5/qa.md:53`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/qa.md#L53)
- **Original Draft**: *"### Q9: Did your magnitude gate beat the conflict gate? What does that prove?"*
- **Audit Assessment**: **VIOLATION OF BANNED TERM RULE**.
- **Resolution**: **REWRITTEN** to:
  *"### Q9: Did your magnitude gate beat the conflict gate? What does that indicate?"*

---

### Hit 10: "proves" / "proving" in Historical Explanation Text
- **Location**: [`outputs/results_explanation.md:138, 160`](file:///Users/jeet/Desktop/AEGIS/outputs/results_explanation.md#L138)
- **Original Draft**: *"This proves that selectively damping weights... proving that the directional conflict check is the source of failure."*
- **Audit Assessment**: **VIOLATION OF BANNED TERM RULE**.
- **Resolution**: **REWRITTEN** to:
  *"This suggests that selectively damping weights... indicating that the directional conflict check is the source of failure."*

---

### Hit 11: Split-MNIST Shared-Head Explanation Labeling
- **Location**: [`README.md`](file:///Users/jeet/Desktop/AEGIS/README.md), [`outputs/phase5/report.md:195`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L195), and summary texts.
- **Original Text**: Marked the explanation that "shared-head binary logits overwrite each other" as `[MEASURED]`.
- **Audit Assessment**: **MISLABELED CLAIM**. The measured data only shows that fine-tuning, EWC, and all gate variants tie between $61.90\%$ and $63.96\%$ ACC on Split-MNIST. Attributing this collapse to output head logit overwrite versus hidden layer representation collapse is a reasoned hypothesis, not directly isolated by the metric.
- **Resolution**: **CORRECTED**. The empirical tie is tagged `[MEASURED]`, while the causal explanation is strictly tagged `[HYPOTHESIS]`.

---

### Hit 12: gate:drift Implementation & Characterization
- **Location**: [`outputs/phase5/report.md:78-81`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L78), [`outputs/phase5/report.md:227`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/report.md#L227), [`outputs/phase5/poster_text.md`](file:///Users/jeet/Desktop/AEGIS/outputs/phase5/poster_text.md).
- **Original Text**: Characterized `gate:drift` as adding an explicit quadratic loss penalty: $\mathcal{L}_{drift}(\theta) = \mathcal{L}_{task}(\theta) + \frac{c}{2}\|\theta - \theta^*\|_2^2$.
- **Audit Assessment**: **INACCURATE FORMULATION**. If an algorithm adds an explicit quadratic penalty to the loss function, it is an L2-SP or EWC regularizer, not a gate. Reading `cl2.py` reveals that `gate:drift` is implemented entirely in `apply(model)` at the gradient level:
  $$\text{cond} = (g \odot d > 0) \lor (g \odot e < 0), \quad e = \theta - \theta_{start}$$
  $$g \leftarrow g \odot (1 - \beta \cdot \text{mag} \cdot \mathbf{1}_{\text{cond}})$$
  It acts as a gradient-domain brake against steps that increase distance from starting weights $\theta_{start}$ ($g \cdot e < 0$).
- **Resolution**: **CORRECTED**. Replaced all references to additive loss penalties with the exact dual-condition gradient damping equation from `cl2.py`, and added an explanation connecting its functional behavior to L2-SP and EWC anchor regularization.

---

### Hit 13: Phase 4 Milestone Status
- **Location**: Project summaries and [`STATUS.md`](file:///Users/jeet/Desktop/AEGIS/STATUS.md).
- **Original Text**: "Every milestone in AGENT_PLAN.md has been executed."
- **Audit Assessment**: **INACCURATE CLAIM**. Phase 4 (GPT-2 language model experiment across 4 domains) was deliberately skipped under the go/no-go gate rule because execution was on local CPU hardware (where training a 124M transformer across 8,000 steps is computationally intractable).
- **Resolution**: **CORRECTED**. Text now explicitly states: Phases 0, 1, 2, 3, and 5 were executed and completed. Phase 4 was skipped per the go/no-go decision rule and is documented strictly as **Future Work**.

---

## 3. Final Certification
All claims in the report, table captions, poster text, viva Q&A, and documentation have been audited and corrected. No banned terms ("prove", "proves", "proven"), unverified assertions, misattributed loss formulas, or overstated milestone claims remain.
