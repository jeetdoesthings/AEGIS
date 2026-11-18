# STATUS.md — research log

Newest entries at the bottom. Commit hashes in brackets refer to `git log`.

---

## Phase 0 — setup (2026-11-09, commit 19df00d)

Froze the original prototype as `cl.py` (never edited after this point) and copied it to `cl2.py`, which carries all further work. Verified the environment: torch 2.8.0 on CPU. Quick benchmark: CPU 0.156 s vs MPS 0.317 s for 500 MLP forward passes — the M4 CPU beats MPS for these tiny matrices, so all runs use CPU. Created the folder layout and this log. Smoke test (`python cl2.py --seeds 1 --epochs 1`) ran in ~16 s.

The smoke test already showed the two problems the semester would be about: EWC scored 9.8% (diverged) and `gate:conflict:1.0` scored 32.4%, while fine-tune/replay/LoRA/magnitude were all 88–93%.

## Phase 1 — making the evaluation honest (commits 2ec680e, c5fb29b, 5457a1c, b1ca604)

**Validation split and seed control (Task 1.1).** Split MNIST train 55,000/5,000 with a fixed seed-42 permutation; added `--eval_split val|test`, `--seed_start`, `--out`, and config logging (all args + git commit + torch version). Sanity check: val ACC 87.2% vs test ACC 87.5% — consistent, and training never sees either eval set.

**EWC repair (Task 1.2).** Diagnosed the 9.80% failure: with λ=5000 and lr 0.05 the quadratic penalty explodes; loss hit NaN at step 13 of task 2 (`outputs/phase1/ewc_diagnosis.txt`), and NaN logits default to class 0, which is exactly 9.80% of the MNIST test set. Added a divergence guard that stops non-finite runs and flags `diverged: true` (failed runs live in [`outputs/failed/`](outputs/failed/README.md)). Swept λ ∈ {1, 5, 20, 100} on validation seeds 0–2; λ=20 won (93.7 ± 0.7% val ACC).

**Fair tuning (Task 1.3).** Swept every method on validation with an equal budget (≤6 configs each; 25 total, `outputs/phase1/tuning_val.csv`). Locked best configs into `configs/locked_perm5.json`. Notable: dropping β to 0.25 rescues gate:conflict from total collapse (82.9% vs ~14.5% at β=1.0) — still 7.9% below fine-tuning.

**Final Perm5 run (Task 1.4).** Locked configs, 5 fresh test seeds 100–104 (`outputs/phase1/final_perm5.json`). Replay best (95.85 ± 0.11), then LoRA (94.30 ± 0.12, BWT exactly 0 via adapters), then EWC (93.76 ± 0.36). gate:conflict is −7.24 ± 1.38 worse than fine-tuning. The prototype's test-set-tuned numbers are now superseded.

## Phase 2 — harder benchmarks (commits f5a03e6, 9eac5c3)

**Perm10 (Task 2.1).** With 10 tasks, fine-tuning's forgetting doubles (BWT −20.80 ± 1.84, ACC 78.21 ± 1.68); gate:conflict collapses to 41.92 ± 1.71. EWC holds 90.22 ± 0.87; gate:magnitude separates from fine-tuning (+4.92 ± 1.20 ACC).

**Split-MNIST (Task 2.2).** Added a domain-incremental benchmark: five binary digit-pair tasks sharing one 2-way head. Tuned and locked (`configs/locked_split.json`, λ=100 for EWC here). Result: every method without replay or a task ID ties at 61.90–63.96% ACC with BWT around −44 to −46; replay (96.85) and LoRA (98.73) are untouched. Cross-benchmark ranking analysis written into STATUS notes and the report.

## Phase 3 — why it failed, and repairs (commit 00453dc)

**Diagnostics (Tasks 3.1–3.2).** `diagnostics.py` logs per-step weight norms, drift along `d`, and gated fraction for fine-tune, gate:conflict, gate:magnitude (3 seeds 0–2, `outputs/phase3/diag_all.csv`). By task 5, layer-1 weight norm under gate:conflict reaches 22.700 vs 11.790 (fine-tune) and drift along `d` is +2.263 vs −0.145 — the asymmetric-ratchet signature. The hypothesis is *supported* (not proven): correlational telemetry plus an intervention.

**Repaired gates (Task 3.3).** Added `drift` (brakes updates that expand distance from the task-start weights or oppose `d`), `hybrid` (conflict × magnitude), and `conflict_soft` (small β). Tuned on Perm5 validation (`configs/locked_repaired.json`), evaluated on all three benchmarks. Perm10: hybrid 83.82 ± 0.54 and drift 83.21 ± 1.26 vs fine-tuning 78.21 ± 1.68 — non-overlapping intervals, paired differences positive on all five seeds. Perm5: drift is borderline (+2.92 ± 0.96; margin of 0.04 points, a tie under the sample-std convention). Split-MNIST: still a tie.

**Layer analysis (Tasks 3.4–3.5).** Gating is heaviest in the classification layers (l3.weight 87.36% gated) and lightest in layer 1 (78.50%) — hypothesis: the gate protects decision boundaries more than generic features.

**Phase 4 — skipped.** The plan's go/no-go rule says run the GPT-2 experiment only on GPU. This machine is CPU-only, so Phase 4 is documented as future work (`PLAN.md`; design in the report §8, including a `GatedAdam` that gates Adam's final update, since a gradient gate is cancelled by Adam's moment normalization). No partial LM results exist.

## Phase 5 — deliverables (commits 572c750, 7e9076c, 65471cd, e3af034, 652d30f, 2f64f81)

Built the results pack (10 methods × 3 benchmarks in `outputs/phase5/`), wrote the paper draft `report.md`, poster text, 60-second script, Q&A, and captions. Ran a claims audit against the raw numbers: replaced "proved" with "supports", labeled the Split-MNIST head-interference explanation as HYPOTHESIS (the tie itself is MEASURED), corrected the drift-gate description (gradient brake, not a loss penalty), and added the honest comparative framing: EWC > every gate on Perm5 and Perm10; gates > fine-tuning by ~5 points on Perm10 only.

## Pre-review audit (2026-11-18, commits up to sem1-review)

Ran a full audit with fresh eyes before the review. Every headline number was recomputed from the raw `results_*.json` files — zero mismatches (`outputs/audit/recomputed_numbers.csv`). Claims ledger: 60 claims, 41 verified, 17 fixed (`outputs/audit/claims_ledger.csv`). Six blockers fixed (wrong lr/batch in docs; conflict_soft β=0.05 and hybrid γ=0.5 labels; magnitude formula vs code; ">80% gated" → measured ~48%; task-1 retention 13.44 ± 4.44; explicit EWC-beats-gates statement for Perm5). Wrote 15 tests in `tests/test_cl2.py` — all pass, including gate mechanics on synthetic tensors, metric identities, split/seed isolation, determinism, β=0 ≡ fine-tuning end-to-end, and a cl.py-vs-cl2.py regression (90.8 ± 2.4 vs 91.3 ± 1.8, within one std). Verified all 14 citations by search (fixed the Titans arXiv ID; removed a propagated wrong L2-SP ID). Verdict: GO.

## Housekeeping (2026-11-18)

Reorganized the repository for the review: Phase-0 prototype artifacts moved to [`archive/phase0/`](archive/phase0/) with a superseded notice, PDF tooling to [`scripts/`](scripts/), the diverged EWC run to [`outputs/failed/`](outputs/failed/README.md), canonical overview documents to [`outputs/final/`](outputs/final/) (byte-identical duplicates removed), empty `lm/` and `phase4/` directories removed, and every absolute `file:///Users/...` link replaced with a relative path. Frozen review deliverables live in `outputs/final/`; tagged `sem1-review` and pushed. A second copy of `outputs/final/` is kept outside the repo.

## Where this is going (next semester)

SI, MAS, A-GEM and L2-SP baselines; gate+EWC; 10 seeds; Split-CIFAR; then the GPT-2 experiment with `GatedAdam` if the vision results justify it; theory of the ratchet; thesis. The full week-by-week plan with decision gates is in `outputs/final/PROJECT_REPORT.md` §13. Standing rules: tune on validation only, 5–10 seeds, equal budgets, claims ledger kept current, no result without a config file.
