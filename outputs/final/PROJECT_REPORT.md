# PROJECT_REPORT.md — AEGIS: Continual Learning with Displacement-Based Update Gating

**Audience:** faculty examiners, and the project's future self.
**Date:** 2026-11-18 · **Repo:** `github.com/jeetdoesthings/AEGIS` · **Tag:** `sem1-review`
Every number cites a raw result file or a table derived directly from one (`outputs/audit/recomputed_numbers.csv`). Claims are labeled MEASURED (read from a result file) or HYPOTHESIS (reasoned, not isolated by these experiments). The project does **not** claim to solve catastrophic forgetting.

---

## 1. Executive summary

**Problem.** When a neural network trained on earlier data is trained on new data, the new gradient updates overwrite the weights that encoded the earlier knowledge, and performance on earlier tasks drops sharply. This is catastrophic forgetting. It has been known since the late 1980s and is unsolved under realistic conditions — no task boundaries given, no unlimited stored data.

**Idea under test.** A family of **update-level gates**: after each task, store `d`, the decayed net movement of each weight during that task (memory ≈ one copy of the weights, no old data). During training on a new task, scale each weight's gradient update using `d`, under plain SGD.

**What was done.** A fair testbed on Permuted-MNIST (5 and 10 tasks) and domain-incremental Split-MNIST with an MLP trained from scratch; baselines (fine-tuning, EWC, replay, LoRA); validation-only tuning with equal budgets; final runs on 5 held-out test seeds (100–104).

**Main findings (MEASURED).**
1. The original conflict gate **failed**: worse than fine-tuning on Perm5 (−7.24 ± 1.38 ACC) and catastrophic on Perm10 (41.92 ± 1.71 vs 78.21 ± 1.68). Step-level telemetry **supports** an asymmetric-ratchet explanation.
2. Repaired variants (`magnitude`, `drift`, `hybrid`) beat fine-tuning on Perm10 by about 5 points with consistent per-seed paired differences — a **modest** gain.
3. **EWC beats every gate variant** on Perm5 (93.76 vs ≤91.89) and Perm10 (90.22 vs ≤83.82) while also storing no old data. This is the project's most important honest finding.
4. On Split-MNIST, no method without replay or a test-time task ID beats 62–64% ACC.
5. Replay and LoRA lead everywhere, with caveats (replay stores data; LoRA uses an oracle task ID).

**What the project does not claim:** no solution to forgetting, no novelty claim for the gate (explicit disclaimer), no statistical significance testing, no language-model evidence yet.

**Next steps:** more baselines (SI, MAS, A-GEM, L2-SP), gate+EWC combinations, a GPT-2 experiment with a gate acting on Adam's final update (details §13).

---

## 2. The problem

**Analogy.** Imagine a musician who practices piano for a year, then takes up guitar. Every hour of guitar practice rewires the same motor circuits the piano skill lives in. Without dedicated protection — or occasional piano practice — a year later the piano skill is gone. Gradient descent is that rewiring: it has no mechanism that says "this weight matters for something learned earlier."

**Technically.** Training sequentially on tasks t = 1…T with objective L_t moves parameters θ to minima of each new task. Because SGD only sees the current loss, nothing prevents a step that degrades performance on earlier tasks. The R-matrix view: R[i][j] = test accuracy on task j after training through task i. Forgetting is the drop in R[i][j] as i grows past j; catastrophic when sharp.

## 3. Why it matters (motivation, not a result)

Deployed models need updating as data and domains change. Retraining from scratch is expensive; naive further training erases earlier skills. Methods that reduce forgetting **without storing old data** matter when old data cannot be kept — privacy, size, licensing. This section is motivation; the experiments above are the evidence.

**Why RAG and long context windows do not address it:** retrieval and long prompts supply text at question time and never change the model's weights. The model recalls facts but acquires no new skill; once the context is gone, nothing persists. Continual learning is about changing the weights safely; these tools are about conditioning on extra text.

## 4. Background and related work

All citations verified by search (arXiv IDs/DOIs in `report.md` References and `AUDIT_REPORT.md` §10).

| Method | stores data? | needs task boundaries? | changes base model? | relative cost |
|---|---|---|---|---|
| EWC (Kirkpatrick 2017) | no | yes (task ends) | yes (anchors) | Fisher + one weight copy |
| Synaptic Intelligence (Zenke 2017) | no | yes | yes | online importance + weight copy |
| MAS (Aljundi 2018) | no | yes | yes | similar to SI |
| LwF (Li & Hoiem 2018) | no | yes | yes | extra distillation forward pass |
| GEM (Lopez-Paz & Ranzato 2017) | **yes** (episodic memory) | yes | yes | QP per step |
| A-GEM (Chaudhry 2019) | **yes** (small) | no | yes | 1 memory pair per step |
| PCGrad (Yu 2020) | no | n/a (multi-task) | yes | gradient projection |
| L2-SP (Li 2019) | no | yes | yes | one weight copy |
| Experience replay (Robins 1995) | **yes** | no | yes | buffer storage + extra loss |
| LoRA (Hu 2021) | no | **yes at test (oracle ID)** | adds adapters | small adapters per task |
| Titans (Behrouz 2024) | partial (internal memory) | no | yes | memory module |
| **Gate (this project)** | no | yes (task end for `d` update) | yes | one weight copy for `d` |

Closest relative: SI — it also derives protection from observed weight movement, but converts it into a quadratic loss penalty rather than a direct gradient gate. The gate is **not novel** (explicit disclaimer in `report.md` §2); what this semester adds is a fair comparison, a diagnosed failure mode, and tested repairs.

## 5. Method

Setup: θ ∈ R^P; at the start of task t, θ_start is frozen. `d` is the decayed sum of per-task net movements (`Gate.end_task`, `cl2.py:106-108`): d ← decay·d + (θ − θ_start). During the task, for each mini-batch gradient g (`Gate.apply`, `cl2.py:110-132`):

- **conflict** (original): g ← g ⊙ (1 − β·mag·1[g·d > 0]), mag = clamp(|d|/(mean|d|+ε), max 1). Blocks only reversals of past movement. `cl2.py:117-119`.
- **magnitude** (ablation): g ← g ⊙ (1 − β·mag), ignoring direction. `cl2.py:120-121`.
- **layer_cos**: per-tensor cosine between g and d; scale the whole tensor's gradient when they align. `cl2.py:122-125`.
- **drift** (repair): g ← g ⊙ (1 − β·mag·1[(g·d>0) ∨ (g·e<0)]), e = θ − θ_start. A **gradient-level brake** — not a loss penalty — damping steps that expand distance from the task anchor or oppose d. Differs from EWC (Fisher-weighted quadratic loss) and L2-SP (isotropic quadratic loss) in acting directly on the gradient, movement-weighted, with no stored Fisher. `cl2.py:126-129`.
- **hybrid** (repair): (1 − β·mag·1[g·d>0]) · (1 − γ·mag). `cl2.py:130-132`.
- **conflict_soft**: conflict with conservative β=0.05.

**SGD requirement.** Gating the gradient scales the SGD step exactly (Δθ = −η·g_gated). With Adam, per-coordinate normalization by √v̂ rescales damped coordinates back up, cancelling the gate; a language-model version must therefore gate Adam's **final update** (`GatedAdam`, `report.md` §8.2).

## 6. Experimental setup

- **Data:** MNIST via torchvision (60k train / 10k test; downloaded to `data/`, not committed). Deterministic seed-42 split: 55k train / 5k validation (`cl2.py:35-44`). Test set touched only by final runs.
- **Benchmarks** (`cl2.py:203-218`): Permuted-MNIST 5 and 10 tasks (per-task fixed pixel permutation, seed = run seed; 10-way head, normalization (x/255 − 0.1307)/0.3081); Split-MNIST domain-incremental: 5 binary tasks (0,1)…(8,9), shared 2-way head, labels re-mapped per task.
- **Model** (`cl2.py:69-81`): MLP 784→256→256→C, ReLU, trained from scratch, no pretrained weights; 269,322 parameters (C=10).
- **Optimizer/training:** plain SGD, lr 0.05, batch 128, 3 epochs/task — **fixed across all methods; lr not tuned** (run scripts). Divergence guard halts non-finite runs (`cl2.py:181-182, 239-248`).
- **Protocol:** tuning on validation only, seeds 0–2, ≤6 configs per method (finetune 1, ewc 4, replay 3, lora 3, conflict 6, magnitude 6, layer_cos 2; split mirrors; repaired: drift 6, hybrid 6, conflict_soft 4 — tuned on Perm5 validation and transferred to the other benchmarks). Best config locked into `configs/locked_*.json` by validation ACC (ties → BWT → NEW), then 5 final test seeds 100–104.
- **Metrics** (§1.9 of the brief; formulas in `report.md` §4.4): ACC, BWT, FWT, NEW; paired per-seed differences vs fine-tuning.
- **Hardware/runtimes:** Apple M4 CPU only (no GPU). Smoke test ≈ 15 s; a 5-seed method-benchmark run is minutes (CPU). MPS was slower for these small MLPs (STATUS.md Task 0.1).
- **Statistics:** mean ± population std (ddof=0) over 5 seeds; tie = overlapping mean±std intervals; "improves" additionally requires consistent paired-difference signs. No significance tests claimed.

## 7. What was done (chronology)

1. **Topic choice.** Continual learning chosen from several candidates as a real open problem suited to small-scale experiments.
2. **Prototype `cl.py`** (frozen, commit 3f2f6ba): Permuted-MNIST 5 tasks, MLP, SGD lr 0.05/batch 128/3 epochs, methods finetune/EWC/replay(200)/LoRA/gates. Flaw: 3 seeds, tuned by looking at test accuracy.
3. **First results (historical, superseded):** finetune 91.57±1.49; replay 95.19±0.09; lora 93.79±0.12; ewc 9.80±0.00; gate:conflict:1.0 15.89±0.41; gate:magnitude:1.0 92.36±0.32.
4. **EWC failure found:** exactly 9.80% at λ ∈ {1000, 5000, 10000} — the class-0 share, consistent with NaN logits defaulting to class 0. Later formally diagnosed: quadratic-penalty explosion to NaN at step 13 (`outputs/phase1/ewc_diagnosis.txt`); repaired with λ=20 (perm) / λ=100 (split) chosen on validation + divergence guard.
5. **Conflict gate failure found:** worse than fine-tuning at every tuned β. Ratchet hypothesis formed (§9).
6. **AGENT_PLAN.md execution** (commits 19df00d…572c750): validation split + seed control (2ec680e), EWC repair (c5fb29b), fair tuning (5457a1c), final Perm5 (b1ca604), Perm10 (f5a03e6), Split-MNIST (9eac5c3), diagnostics + repaired variants + layer analysis (00453dc), deliverables pack (572c750), wording fixes (7e9076c, 652d30f). Phase 4 (GPT-2) **not done** — skipped per go/no-go rule (CPU-only machine), documented as future work.
7. **This audit** (2026-11-18): recomputation of all numbers from raw files, 15 tests, 6 Blocker fixes, reports, tag `sem1-review` (see `AUDIT_REPORT.md`).

## 8. Results

Full tables with FWT/NEW and paired differences: `report.md` §5, `outputs/phase5/table_*.csv`, `outputs/phase5/master_summary.csv`. Summary (ACC %, mean ± std over 5 test seeds 100–104, ddof=0; source JSONs in §5 of AUDIT_REPORT):

| Method | Perm5 | Perm10 | Split |
|---|---|---|---|
| Fine-tuning | 88.97 ± 1.85 | 78.21 ± 1.68 | 62.07 ± 1.00 |
| EWC (λ 20/100) | **93.76 ± 0.36** | **90.22 ± 0.87** | 63.96 ± 1.39 |
| Replay (1000/task) | 95.85 ± 0.11 | 95.22 ± 0.09 | 96.85 ± 0.30 |
| LoRA (oracle ID) | 94.30 ± 0.12 | 94.05 ± 0.05 | 98.73 ± 0.06 |
| Gate: conflict (β 0.25) | 81.73 ± 1.41 | 41.92 ± 1.71 | 62.28 ± 0.98 |
| Gate: magnitude (β 1.0) | 91.43 ± 0.86 | 83.13 ± 1.22 | 61.90 ± 0.36 |
| Gate: layer_cos | 89.00 ± 2.32 | 78.80 ± 0.97 | 62.05 ± 1.02 |
| Gate: drift (β 0.1) | 91.89 ± 1.03 | 83.21 ± 1.26 | 62.62 ± 1.03 |
| Gate: hybrid (β 0.1, γ 0.5) | 90.98 ± 0.90 | 83.82 ± 0.54 | 62.78 ± 0.56 |
| Gate: conflict_soft (β 0.05) | 90.68 ± 0.58 | 80.05 ± 1.66 | 62.34 ± 1.03 |

- **Paired differences vs fine-tuning** (same seed, same permutations; mean ± std of the 5 per-seed differences): Perm10 — magnitude +4.92±1.20, drift +5.01±1.48, hybrid +5.62±1.21, all positive on 5/5 seeds with non-overlapping intervals → genuine, **modest** gains. Perm5 — drift +2.92±0.96 (borderline: 0.04-point interval margin; a tie under ddof=1), magnitude +2.46±1.73 (signs +5/5, intervals overlap → tie), hybrid/conflict_soft ties. Split — all ties.
- **Forgetting curves and heatmaps:** `outputs/phase5/forgetting_curve_*.png`, `heatmaps_*.png` (task-1 retention: conflict 13.44±4.44 after 10 tasks vs finetune 47.31±4.80).
- **Plasticity–stability view (NEW vs BWT):** conflict keeps NEW high (94.86±0.15 on Perm10) while BWT collapses (−58.83) — it forgets, it does not refuse to learn. LoRA (NEW = final ACC, BWT 0) and replay dominate the frontier; gates shift BWT from −20.8 to about −13…−15 at a cost of ~2 NEW points on Perm10.
- All statements above are MEASURED from the cited files under the tie rule.

## 9. Analysis

**The ratchet diagnosis (supported, not proven).** Correlational evidence: under gate:conflict, layer-1 weight norms grow to 22.700 by task 5 (vs 11.790 finetune, 12.291 magnitude) and drift along d is +2.263 (vs −0.145, −0.016) — 3 diagnostic seeds 0–2 (`outputs/phase3/diag_all.csv`); ~48% of elements are gated per step. Interventional evidence: repairs that remove the asymmetry (drift, hybrid) eliminate the runaway and recover ~5 points on Perm10. **Alternatives not excluded:** (a) gated updates might simply slow new-task learning — partly addressed by NEW, which stays high for conflict, so the failure is retention, not plasticity; (b) d may be dominated by task 1's movement away from random initialization, making the protected direction an initialization artifact. Distinguishing (b) needs re-anchoring or per-task displacement ablations (next semester).

**Why EWC beats the gates (HYPOTHESIS).** EWC's quadratic penalty is exactly proportional to displacement from an anchor with per-weight Fisher weighting; the gates apply a bounded, saturating scale (mag clamped at 1, factor ≤ 1−β) that cannot grow with displacement, and conflict/drift still permit small outward steps. EWC also directly protects the output head, which matters as tasks accumulate.

**Why nothing buffer-free works on Split-MNIST (HYPOTHESIS).** The shared 2-way output head is overwritten by each new digit pair; weight-space regularization cannot resolve direct logit interference at the head. The MEASURED fact is the tie at 61.90–63.96; the head-interference mechanism is not isolated by these metrics (per-unit head telemetry would be needed).

**Why FWT ≈ 0 on permuted tasks.** Each task is an independent random pixel permutation; prior tasks confer no head start, so accuracy on an unseen task matches the untrained reference (FWT 0.47–2.05 across methods, `final_perm*_table.csv`). On Split-MNIST the reference is the untrained 2-way head (~50%).

## 10. File-by-file guide

| path | type | what it is | produced by | used by | why it matters | status |
|---|---|---|---|---|---|---|
| `cl.py` | code | frozen original prototype (never edited) | initial commit | regression tests | reference implementation | final |
| `cl2.py` | code | main engine: splits, seeds, divergence guard, diagnostics hooks, 6 gate modes, benchmarks | Phase 0+ | all run/tune scripts, tests | single source of truth for methods | final |
| `tune_val.py` | code | Perm5 validation sweep (25 configs, seeds 0–2) | Task 1.3 | configs/locked_perm5.json | fair tuning | final |
| `run_final_perm5.py` | code | 5-seed test evaluation, Perm5 | Task 1.4 | outputs/phase1/final_* | headline table | final |
| `run_perm10.py` | code | 5-seed test evaluation, Perm10 | Task 2.1 | outputs/phase2/final_* | long-horizon table | final |
| `tune_and_run_split.py` | code | tune + test Split-MNIST | Task 2.2 | outputs/phase2/final_split* | domain-incremental table | final |
| `tune_and_run_repaired.py` | code | tune + test drift/hybrid/conflict_soft on all 3 benchmarks | Task 3.3 | outputs/phase3/final_repaired_* | repaired-variant tables | final |
| `diagnostics.py` | code | step-level norms, gated fraction, drift along d; ratchet test | Task 3.1-2 | outputs/phase3/diag_*.csv | ratchet evidence | final |
| `layer_analysis.py` | code | per-layer gated-fraction heatmap | Task 3.4 | outputs/layer_protection.png, phase3 heatmap | layer hierarchy | final |
| `build_results_pack.py` | code | consolidates tables/plots into outputs/phase5/ | Task 5.1 | phase5 pack | deliverables | final |
| `build_master_pdf.py`, `build_simplified_pdf.py` | code | build the two overview PDFs | post-Phase 5 | outputs/AEGIS_*.pdf | overview docs | final |
| `analyze.py` | code | early plotting/summary script | Phase 0 era | outputs/*_main etc | historical | superseded |
| `run_sweep.sh` | script | early gate sweep runner | Phase 0 era | outputs/gate_sweep | historical | superseded |
| `tests/test_cl2.py` | code | 15 audit tests (gate mechanics, metrics, splits, seeds, determinism, EWC, replay, β=0≡FT) | this audit | pytest | correctness guarantees | final |
| `configs/locked_{perm5,split,repaired}.json` | config | locked best hyperparameters per benchmark | tune scripts | run scripts | tuning/test isolation | final |
| `logs/` | logs | run logs incl. `audit_slow_tests.log` | runs | audit | provenance | final |
| `outputs/phase1/` | results | EWC diagnosis + λ sweep, Perm5 tuning + final tables/curves/heatmaps, diverged-run record | Tasks 1.1–1.4 | report | Perm5 evidence | final |
| `outputs/phase2/` | results | Perm10 + Split tuning/final tables, curves, heatmaps | Tasks 2.1–2.2 | report | Perm10/Split evidence | final |
| `outputs/phase3/` | results | diagnostics CSVs, ratchet hypothesis/findings, repaired tables, layer heatmap | Tasks 3.1–3.5 | report | mechanism + repairs | final |
| `outputs/phase4/`, `lm/` | — | empty (Phase 4 not done) | — | — | proof Phase 4 not run | scratch |
| `outputs/phase5/` | results | report.md, claim_audit.md, poster_text.md, script_60s.md, qa.md, captions.md, master_summary.*, table_*, curves, heatmaps, diagnostics, layer heatmap; plus copies of the two overview docs | Task 5.x | presentation | deliverable pack | final |
| `outputs/final/` | results | AUDIT_REPORT.md, PROJECT_REPORT.md (+PDFs), poster, script, qa, master_summary.csv, figures | this audit | the review | frozen deliverables | final |
| `outputs/audit/` | results | recompute.py, recomputed_numbers.csv, claims_ledger.csv, regression artifacts | this audit | AUDIT_REPORT | evidence of verification | final |
| `outputs/AEGIS_Master_Project_Report.*`, `outputs/AEGIS_Simplified_Project_Overview.*` | docs | long/short overviews (PDF+MD) | post-Phase 5 | examiners | consistent with tables | final |
| `outputs/summary.md`, `outputs/results_explanation.md` | docs | prototype-era summaries (test-set-tuned numbers) | Phase 0 era | — | historical only | **superseded** |
| `STATUS.md` | doc | milestone log with per-task numbers | all phases | audit trail | chronology evidence | final |
| `AGENT_PLAN.md` | doc | the executed plan (Phases 0–5, cut rules) | pre-Phase 0 | STATUS | scope reference | final |
| `MASTER_BRIEF.md` | doc | this audit's instructions (untracked/added by user) | user | audit | audit spec | final |
| `README.md`, `requirements.txt` | docs | install, smoke test, reproduction commands, honest results table | Tasks 5.5 / audit | reproduction | entry point | final |
| `.gitignore` | config | excludes `.venv/`, `data/`, `__pycache__/`, OS files | initial | git | hygiene | final |

## 11. Limitations

1. Small MLP (269k params) on MNIST variants; no convolutions or transformers.
2. Five test seeds; seeds also change the task permutations (std includes task-sequence variance).
3. One optimizer (plain SGD); no Adam evidence.
4. Limited tuning budget (≤6 configs per method); repaired gates tuned on Perm5 validation only, transferred to Perm10/Split.
5. LoRA uses an oracle task ID at test time.
6. **EWC beats every gate variant on Perm5 and Perm10** while also storing no old data.
7. No gain on Split-MNIST for any buffer-free method.
8. No language-model evidence yet.
9. Limited headroom: fine-tuning already reaches 78–89% on the permuted benchmarks.
10. The first prototype experiments tuned on the test set; corrected in the final protocol (kept as a recorded flaw).
11. No significance testing; 18 comparisons made, some chance findings expected.
12. CPU-only hardware; no GPU cross-check.

## 12. Contribution statement

This semester's contribution is: **(1)** a fair, reproducible testbed for update-level gating on three continual-learning benchmarks with validation-only tuning, equal budgets, locked configs, and held-out test seeds; **(2)** a diagnosed failure — the original conflict gate's asymmetric-ratchet collapse, supported by step-level telemetry and an intervention (repair) experiment, with named alternative explanations; **(3)** repaired variants (magnitude, drift, hybrid) with a modest, seed-consistent gain over fine-tuning on the 10-task benchmark; and **(4)** an honest comparison showing EWC — itself buffer-free — beats every gate variant on both permuted benchmarks, and that no buffer-free method helps on Split-MNIST. The project does not claim to solve catastrophic forgetting.

## 13. Future work (next semester, week by week)

- **Weeks 1–3:** add SI, MAS, A-GEM, L2-SP baselines; test gate+EWC combination; study why EWC wins; raise seeds to 10; add Split-CIFAR. *Gate:* if a variant beats EWC, continue the gate line; otherwise pivot to "gate plus EWC" and the diagnosis.
- **Weeks 4–8:** GPT-2 small on four text domains with `GatedAdam` (gate on Adam's final update); compare finetune/replay/EWC/LoRA-with-domain-ID/best gates; perplexity per domain. Go only if the vision study is stable; report negative results as they are.
- **Weeks 9–11:** practical setting with a small open model (~0.5B) on 2–3 tasks; cost and memory audit. Skip if LM results are weak.
- **Weeks 12–14:** formal ratchet explanation; sensitivity to β, decay, lr; ablations; layer analysis.
- **Weeks 15–16:** thesis report and workshop-style paper.
- **Standing rules:** tune on validation only; 5–10 seeds; equal tuning effort per method; claims ledger kept current; no result without a config file.
- **Final goal:** determine whether a cheap, no-storage, update-level gate reduces forgetting and where it sits against EWC, replay and LoRA on the stability–plasticity tradeoff. Success = a variant matching/beating EWC with a language-model gain, **or** a negative result with a clear explanation and a rigorous comparison. The project never claims to solve catastrophic forgetting.

## 14. FAQ

**Common questions**

1. **What is catastrophic forgetting?** A network trained sequentially loses earlier skills because new-task gradients overwrite the weights that encoded them. Measured here as negative BWT: finetune BWT −9.62±2.34 (Perm5), −20.80±1.84 (Perm10).
2. **What is this project, in one minute?** We tested buffer-free gates that scale each weight's update using the weight's past net movement `d`. The original conflict gate failed (ratchet); repairs beat fine-tuning by ~5 points on 10-task Permuted-MNIST; but EWC, also buffer-free, beats all gates (90.22 vs ≤83.82 Perm10 ACC).
3. **Why does it matter?** Deployed models need updates without storing old data (privacy, size, licensing). Motivation only — the evidence is in §8.
4. **Same as RAG / long context?** No. RAG and context windows never change weights; they supply text at question time. They cannot acquire skills.
5. **Datasets?** MNIST (torchvision; 60k train/10k test), split 55k/5k/10k; Permuted-MNIST 5 & 10 tasks, Split-MNIST 5 binary tasks. Small, standard, fully reproducible.
6. **Model? Why so small?** MLP 784-256-256-C, 269,322 params, trained from scratch. Small enough for exhaustive multi-seed sweeps on CPU and for exact weight-space diagnostics.
7. **What are Permuted/Split-MNIST; domain-incremental?** Permuted: same digits, per-task pixel permutations — the task changes, the 10-way head is shared (domain-incremental). Split: five digit-pair binary tasks sharing a 2-way head. van de Ven & Tolias (2019) define the scenarios.
8. **Which methods?** Fine-tuning, EWC, replay (1000/task), LoRA (r=16/8, oracle ID), and six gate variants (conflict, magnitude, layer_cos, drift, hybrid, conflict_soft).
9. **The gate in plain words?** "Weights that moved a lot for old tasks get their updates damped during new tasks; the original version only blocked updates that would undo old movement, which turned out to be the failure."
10. **ACC/BWT/FWT/NEW; why NEW?** ACC = final average accuracy; BWT = average change on old tasks (forgetting); FWT = accuracy on new tasks vs an untrained net; NEW = accuracy right after learning each task. NEW is always reported so a method cannot look good by refusing to learn — and indeed gate:conflict keeps NEW 94.86 while ACC collapses to 41.92.
11. **Main results?** §8's table; key: repairs +4.9…+5.6 over finetune on Perm10; EWC +12.0 over finetune; replay/LoRA ≫ all.
12. **Did the idea work?** Partially: the original gate failed; the repairs give a modest, seed-consistent gain over fine-tuning on Perm10 only. The strongest buffer-free baseline (EWC) beats them.
13. **Why did the conflict gate fail?** Supported mechanism: asymmetric ratchet — blocks reversals, permits outward steps; norms doubled (22.70 vs 11.79), drift along d +2.263. Alternatives not excluded (§9).
14. **EWC, and why did it fail at first?** Fisher-weighted quadratic anchor penalty. First runs used λ=5000 with lr 0.05 → penalty explosion → NaN → 9.80% (class-0 share). Diagnosed (`ewc_diagnosis.txt`), guarded, repaired with λ=20/100 chosen on validation.
15. **Replay?** Stores 1000 real examples per task and interleaves them into training; best ACC on Perm5/Perm10. Its caveat: it stores data.
16. **LoRA, and why unfair?** Per-task low-rank adapters; BWT exactly 0. Unfair without a caveat: choosing the adapter needs the task ID at test time, which task-free continual learning does not assume.
17. **How many seeds, why?** Tuning seeds 0–2; final seeds 100–104 (disjoint). Five is small; intervals are wide; conclusions are kept inside the tie rule.
18. **Tuning fair?** Equal ≤6-config budgets on the validation split only; configs locked before test runs; the prototype's test-set tuning was the recorded flaw the final protocol corrected.
19. **What did you build; which files?** §10's table. Core: `cl2.py` (engine), `tune_*`/`run_*` (protocol), `tests/` (15 passing tests), `configs/` (locked configs), `outputs/phase1-3,5` (results).
20. **Next semester?** §13: more baselines, gate+EWC, GPT-2 with GatedAdam, theory, thesis.

**Complex questions**

1. **How do you know the ratchet explanation is right?** Two evidence types (correlational logs + interventional repair) support it; alternatives — plasticity loss, initialization-dominated `d` — are not excluded (§9). "Supported", not proven.
2. **Is `drift` just EWC/L2-SP?** It is a gradient-level, movement-weighted brake with no Fisher estimate and no loss term; L2-SP/EWC add quadratic loss penalties. Functionally anchored, mechanically different; the report states this.
3. **Relation to SI/GEM/A-GEM/PCGrad/MAS/LwF; is it new?** The gate is not novel (explicit disclaimer): SI tracks movement, GEM/A-GEM use gradient conflict with memory, PCGrad projects conflicting gradients, MAS/LwF are other protection routes. The contribution is the diagnosis and fair comparison.
4. **Why SGD? Adam? LMs?** Gradient scaling = step scaling only under SGD. Adam's √v̂ normalization cancels gradient gates; `GatedAdam` (report §8.2) gates the final update instead — the planned LM method.
5. **Why displacement, not stored gradients/Fisher?** Displacement costs one weight copy, needs no stored data or extra passes, and is measurable online. Fisher (EWC) is per-weight importance but no direction; stored gradients (GEM) need memory. Head-to-head costs: §4 table.
6. **5 seeds enough? Intervals? Multiple comparisons?** No — five is minimal; that is why only interval overlap + consistent paired signs are claimed, no significance tests, and a multiple-comparisons note is included (report §4.5). Next semester: 10 seeds.
7. **Leakage prevention; what went wrong first?** Tuning on validation seeds 0–2 only; locked JSONs; fresh test seeds 100–104. What went wrong first: the prototype tuned by test accuracy — an acknowledged flaw, corrected.
8. **Why does EWC win; what would theory look like?** HYPOTHESIS: unbounded, Fisher-weighted anchor penalty vs bounded saturating gate scale (§9). Theory: model the gate as anisotropic damping of an OU process around the anchor and compare stationary forgetting rates.
9. **Why does nothing buffer-free work on Split-MNIST?** MEASURED: 61.90–63.96 tie. HYPOTHESIS: shared-head logit overwrite (§9). What would help: per-task heads, replay, routing — replay (96.85) and LoRA (98.73) confirm.
10. **Why is FWT ≈ 0?** Random permutations give no transferable features (§9).
11. **Plasticity–stability positions (NEW vs BWT)?** §8 bullet: conflict high-NEW/collapse-BWT; gates trade ~2 NEW for ~6–8 BWT points on Perm10; EWC shifts further; LoRA/replay at the stability extreme.
12. **Memory costs?** Gate: one weight copy for `d` (≈1.08 MB fp32). EWC: Fisher + anchors (two copies). Replay: 1000 images/task (≈0.08 MB fp32 per task pixels + labels) but stores real data. LoRA: adapters per task (r=16: ≈25k params/task).
13. **Would this scale to LMs? Evidence?** No evidence yet — Phase 4 is next semester. The design exists (`GatedAdam`) because Adam cancels gradient gates.
14. **Implementation correctness?** 15 tests: gate mechanics on synthetic tensors, metric identities on hand-computed R, split/seed disjointness, determinism, β=0 ≡ fine-tune end-to-end, EWC finiteness, replay size; plus a cl.py↔cl2.py regression (90.8±2.4 vs 91.3±1.8).
15. **Sensitivity to lr/β/decay?** Partially measured: β grids swept per gate (tuning CSVs); conflict collapses at β=1.0 (15.89±0.41 prototype) and recovers at β=0.25 (82.92±1.79 val). Full sensitivity study is weeks 12–14.
16. **Dependence on task order/permutations?** Seeds change permutations; std includes that variance (B12 note); 5-seed paired diffs are same-seed so permutations are controlled within each comparison.
17. **Why gains on 10 tasks but not clearly on 5?** HYPOTHESIS: with 20.80 BWT points to protect at 10 tasks there is more headroom; on 5 tasks finetune loses only 9.62 and intervals are wide relative to effects.
18. **What would convince you the idea is wrong?** If with proper per-benchmark tuning, more seeds, and an SI/EWC-equivalent budget no gate variant closes the gap to EWC on Perm10/Split-CIFAR, and the ratchet diagnosis fails to explain reproducible failure signatures, the gate line should pivot (week 1–3 gate in §13).
19. **If EWC wins, what is the contribution?** The testbed, the diagnosed failure mode, the repairs' honest quantification, and the EWC comparison itself — a negative/benchmark result with explanation is a result (I7).
20. **Why did prototype and final numbers differ?** Prototype tuned on test, 3 seeds, EWC diverged, conflict at β=1.0; final protocol fixed all four (validation tuning, 5 fresh seeds, λ repair, β sweep).
21. **Replay buffer 1000 — fair to a no-storage method?** It is deliberately generous: replay marks the storage-friendly ceiling, and the gate's claim is only ever relative to fine-tuning and EWC. The asymmetry is stated in every table's caption.
22. **Title mentions conflict gating but that variant failed — why keep it?** The failure **is** the finding: the title names the idea under test, the report documents its failure and repairs. Renaming is queued as a user decision (AUDIT_REPORT §9), with candidate titles offered.

## 15. Glossary

**catastrophic forgetting** — sharp loss of earlier-task performance when training on new tasks. **continual learning** — learning task sequences without full retraining. **domain-incremental** — same output space, shifting input distribution (permuted pixels; shared head here). **task-incremental** — task identity available at test time (LoRA's setting). **class-incremental** — new classes over time, no task ID. **ACC** — mean accuracy on all tasks after the last task. **BWT** — mean change on old tasks after later training (negative = forgetting). **FWT** — accuracy on a new task vs an untrained net. **NEW** — accuracy right after learning each task. **Fisher information** — per-weight sensitivity of the loss; EWC's importance weight. **EWC** — Fisher-weighted quadratic anchor penalty. **replay** — storing and retraining on old examples. **LoRA** — low-rank per-task adapters on a frozen base. **gate** — per-weight scaling of updates. **`d`** — decayed cumulative net weight movement over past tasks. **ratchet** — the failure where blocking only reversals lets weights run away along `d`. **plasticity** — ability to learn new tasks (NEW). **stability** — retention of old tasks (BWT). **seed** — RNG seed; also selects the task permutations here. **validation split** — 5k held-out training images used only for tuning. **tie rule** — overlapping mean±std intervals = tie; "improves" needs non-overlap + consistent paired-difference signs. **oracle task ID** — true task label given at test time. **paired difference** — per-seed (same permutation) difference between a method and fine-tuning.

## 16. Appendix

- **Reproduction:** `README.md` §Reproduction (smoke test: `python cl2.py --seeds 1 --epochs 1`; per-benchmark commands 1–8). Clean-clone result: `AUDIT_REPORT.md` §8.
- **Environment:** Python 3.9.6; torch 2.8.0, torchvision 0.23.0, numpy 2.0.2, pandas 2.3.3, matplotlib 3.9.4 (`requirements.txt`); Apple M4, CPU.
- **Claims ledger:** `outputs/audit/claims_ledger.csv` (60 claims; 41 verified, 17 fixed, 2 cosmetic open).
- **Tuning tables:** `outputs/phase1/tuning_val.csv` (25 configs), `outputs/phase2/tuning_split_val.csv` (25), `outputs/phase3/tuning_repaired_val.csv` (16); locked values in `configs/locked_*.json`.
- **Audit:** `outputs/final/AUDIT_REPORT.md` — verdict GO; all Blockers fixed; tests 15/15 pass.
