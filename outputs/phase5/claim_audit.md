# Claim Audit: Text Verification & Resolution Log

This document audits all occurrences of sensitive epistemological and comparative terms—**"prove"**, **"solve"**, **"solved"**, **"beats"**, **"outperforms"**, **"novel"**, and **"state-of-the-art"** (and their grammatical derivatives)—across the project deliverables in `outputs/phase5/` and related analytical texts.

Every detected instance has been inspected against the raw experimental data files and evaluated under the **Statistical Tie Rule**:
> *Two methods are reported as a statistical tie if their `mean ± std` intervals overlap. Claims of 'beating' or 'outperforming' are prohibited for overlapping intervals and are permitted only when empirical distributions are strictly disjoint.*

---

## 1. Audit Summary Table

| Term Checked | Occurrences Found | Compliant / Supported | Rewritten / Resolved |
| :--- | :---: | :---: | :---: |
| **novel / novelty** | 2 | 2 (Both explicitly disclaim novelty) | 0 |
| **state-of-the-art / sota** | 0 | 0 | 0 |
| **prove / proved / proves** | 0 | 0 | 0 |
| **solve / solved / solves** | 1 | 1 (Contextual usage: task objective) | 0 |
| **beats** | 1 | 1 (Rule definition text only) | 0 |
| **outperform / outperforming** | 6 | 5 (Supported by non-overlapping intervals) | 1 (Rewritten to note statistical tie) |

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

## 3. Final Certification
All claims in the report, table captions, and presentation materials have been fully audited against the generated results. No claims of novelty, unverified superiority, or speculative breakthroughs remain.
