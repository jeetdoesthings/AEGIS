# AUDIT_REPORT.md — AEGIS Pre-Review Audit

**Date:** 2026-11-18
**Auditor:** fresh AI agent session (no prior memory), per `MASTER_BRIEF.md`
**Scope:** this semester's work only (vision testbed, Phases 0–3 and 5; Phase 4 confirmed not done).

---

## 1. Verdict

**GO** for tomorrow's review.

Every headline number in the master results table, the three benchmark tables, and the paired differences reproduces exactly from the raw `results_*.json` files under the population-std convention (ddof=0) — 0 mismatches over 60 checked values (`outputs/audit/recomputed_numbers.csv`). The three known wording problems (A6) were absent or already repaired, and every Blocker found in this audit (six) was fixed today, with all fixes committed before the `sem1-review` tag. The project's claims are now inside its evidence: it reports a diagnosed failure (the conflict gate), modest repaired-variant gains over fine-tuning on Perm10, an explicit statement that EWC beats every gate variant on Perm5 and Perm10, and no claim to solve catastrophic forgetting. The main residual risks are presentation-level, not evidential: no designed poster PDF exists (only poster text), and the repaired-gate hyperparameters were tuned on Perm5 validation and transferred to the other two benchmarks — both are honest answers an examiner can be given.

## 2. Scope and method

**Checked.** (i) All 60 table numbers in brief §1.6 / README / `report.md` recomputed from raw JSON (`outputs/phase1/final_perm5.json`, `outputs/phase2/final_perm10.json`, `outputs/phase2/final_split.json`, `outputs/phase3/final_repaired_{perm5,perm10,splitmnist}.json`) by `outputs/audit/recompute.py`; (ii) per-seed paired differences and tie status for all 9 methods vs fine-tuning on all 3 benchmarks; (iii) diagnostics figures recomputed from `outputs/phase3/diag_all.csv`; (iv) code audit of `cl2.py`, `cl.py`, `diagnostics.py`, `tune_*.py`, `run_*.py`, `layer_analysis.py` against every method description; (v) 15 unit/integration tests written and run (`tests/test_cl2.py`); (vi) config chronology via git log; tuning grids counted from the three `tuning_*.csv` files; (vii) all 14 references verified by web search; (viii) banned-word grep across all tracked markdown; (ix) EWC divergence diagnosis re-read (`outputs/phase1/ewc_diagnosis.txt`); (x) git history scanned for secrets.

**Not checked, and why.**
- **CPU-vs-GPU equivalence (C6):** no CUDA device exists on this machine; all runs are CPU. Recorded as a limitation.
- **Full re-execution of the 5-seed final runs (F1, full form):** re-running all 10 methods × 5 seeds × 3 benchmarks is ~hours of CPU. A clean-clone smoke test and targeted re-runs (regression, determinism, EWC, beta-0 equivalence) were run instead; results below.
- **The two generated overview documents** (`AEGIS_Master_Project_Report.md/pdf`, `AEGIS_Simplified_Project_Overview.md/pdf`) were spot-checked, not re-audited line by line; their tables match `master_summary.csv`.
- **`lm/` and `outputs/phase4/`:** were empty (Phase 4 not done, as the brief instructed to assume); removed during the post-audit cleanup.

## 3. Findings table

| ID | area | severity | finding | evidence | fix | status |
|---|---|---|---|---|---|---|
| F01 | A2 | PASS | all headline numbers reproduce from raw JSON | `outputs/audit/recomputed_numbers.csv` | — | closed |
| F02 | A3 | Major | std convention (numpy ddof=0) was nowhere stated | `run_final_perm5.py` uses `.std()` | convention stated in `report.md` §4.4 and captions; ddof=1 equivalents in §6 here | fixed |
| F03 | A6a | PASS | no "prove/proved/proof" in any deliverable (already purged in commit 7e9076c) | grep across `*.md` | — | closed |
| F04 | A6b | PASS | Split-MNIST shared-head explanation labeled HYPOTHESIS, tie labeled MEASURED | `report.md` §5.3 obs 3; `qa.md` Q12 | — | closed |
| F05 | A6c | Blocker | documents did not state EWC beats every gate on **Perm5** (only Perm10) | README exec summary; `report.md` exec summary 4 | Perm5 statement added to both | fixed |
| F06 | C1 | Blocker | §3.2 magnitude formula said `\|d\|/max(\|d\])` with `max(0,·)`; code uses `\|d\|/mean(\|d\|)` clamped at max 1 | `cl2.py:116` vs `report.md` §3.2 | formula rewritten to match code | fixed |
| F07 | C2 | PASS | `gate:drift` is a gradient-level brake, not a loss penalty; documents describe it correctly | `cl2.py:126-129`; `qa.md` Q10; STATUS Final Verification 3 | — | closed |
| F08 | B9 | Blocker | three documents said lr 0.01 / batch 64; all runs used lr 0.05 / batch 128 / 3 epochs, fixed across methods, lr not tuned | `run_final_perm5.py`, `tune_and_run_split.py:57,149`, `tune_and_run_repaired.py:49,139` | corrected in `report.md` §4.3, `poster_text.md` §2, `qa.md` Q6 | fixed |
| F09 | C1/G7 | Blocker | conflict_soft final runs used β=0.05 but tables labeled β=0.1; poster labeled hybrid γ=0.25 but locked value is γ=0.5 | `configs/locked_repaired.json` vs README/report/poster tables | labels corrected everywhere to locked values | fixed |
| F10 | D1 | Blocker | ">80% of elements gated at every step" contradicts raw data: mean gated fraction ≈ 48% (conflict, tasks 2–5) | `outputs/phase3/diag_all.csv` (0.479) | corrected in `report.md` §6.1 and captions Fig 9 | fixed |
| F11 | A2 | Blocker | `report.md` §5.2 said Task-1 retention 14.28±1.57; raw value is 13.44±4.44 | `outputs/phase2/final_perm10.json` R[-1][0] | corrected | fixed |
| F12 | G3 | Major | Titans cited with wrong arXiv ID (2412.19837 = an unrelated paper) and wrong authors; L2-SP "arXiv:1809.05322" is a propagated citation error (that ID is a robotics paper); MAS, LwF, PCGrad, A-GEM, L2-SP, van de Ven, Robins, Goodfellow missing from references | web-search verification (§10 of this report) | references list rebuilt with verified IDs; wrong IDs removed | fixed |
| F13 | C3 | Major | no test suite existed | repo tree | 15 tests written in `tests/test_cl2.py`; all pass | fixed |
| F14 | F2 | Major | no `requirements.txt` | repo tree | created with exact pinned versions (torch 2.8.0, torchvision 0.23.0, numpy 2.0.2, pandas 2.3.3, matplotlib 3.9.4; Python 3.9.6) | fixed |
| F15 | A8 | Major | historical prototype-era docs (`archive/phase0/summary.md`, `archive/phase0/results_explanation.md` after today's reorganization; originally `outputs/summary.md`, `outputs/results_explanation.md`) carried old test-set-tuned numbers with no marker | file contents | SUPERSEDED banner added to both | fixed |
| F16 | E2 | Major | "statistically outperform(s)" used in 6 places with no named test | grep | reworded to "outperforms under the tie rule (non-overlapping intervals, paired signs consistent)"; report §4.5 now states no significance test is claimed | fixed |
| F17 | I3 | Major | Perm5 `gate:drift` "outperforms" was stated without the borderline caveat: margin is 0.04 points and the result is a tie under ddof=1 | `outputs/audit/recomputed_numbers.csv` | borderline caveat added in `report.md` (exec summary + §5.1) and `qa.md` Q10 | fixed |
| F18 | D2 | Major | ratchet section named neither evidence type nor alternative explanations | `report.md` §6.1 | rewritten: correlational logs + interventional repair evidence; two named alternatives (plasticity loss; d dominated by task-1 movement) with the NEW-metric counterpoint | fixed |
| F19 | D3 | Major | diagnostics seed count (3 seeds, 0–2) unstated in report/captions | `diagnostics.py:16` | stated in §6.1, captions Figs 7–9 | fixed |
| F20 | B11 | Major | no one-sentence FWT≈0 explanation in report; Split FWT reference unexplained | `report.md` §5 intro | sentence added (permuted tasks give no head start; Split reference = untrained 2-way head) | fixed |
| F21 | C4 | PASS | frozen `cl.py` vs `cl2.py` finetune regression: 90.8±2.4 vs 91.3±1.8 ACC (seeds 0–2), intervals overlap | `outputs/audit/regression/` | — | closed |
| F22 | B1 | PASS | train/val/test disjoint by construction (seed-42 randperm, 55k/5k + 10k official test); tuning scripts pass `eval_split="val"` only; "best" selection reads validation ACC from tuning CSVs | `cl2.py:35-44`; `tune_val.py`; test `test_train_val_test_disjoint` | — | closed |
| F23 | B2 | PASS | tuning seeds {0,1,2} disjoint from final seeds {100..104} | `tune_*.py`; test `test_seed_sets_disjoint` | — | closed |
| F24 | B3 | partial | `locked_perm5.json` committed (5457a1c) before final perm5 run commit (b1ca604); `locked_split.json` and `locked_repaired.json` were written by their tune-and-run scripts in the same commit as the final outputs (git cannot order within a commit) | `git log -- configs/ outputs/` | accepted; script flow locks configs before test evaluation | closed (partial evidence) |
| F25 | B4 | PASS | configs per method on validation ≤6: finetune 1, ewc 4, replay 3, lora 3, conflict 6, magnitude 6, layer_cos 2 (=25); split same (=25); repaired drift 6, hybrid 6, conflict_soft 4 (=16) | `outputs/phase*/tuning_*.csv` | — | closed |
| F26 | B5 | PASS | no NaN/diverged entry in any final table; λ=20 (perm) and λ=100 (split) chosen on validation; Fisher estimated on the just-finished task's training data (`cl2.py:252`, Xp = current-task train split); `ewc_diagnosis.txt` exists | `outputs/phase1/ewc_diagnosis.txt`; `results_ewc_val_lam*.json` | — | closed |
| F27 | B6 | PASS | diverged run (`results_ewc_test_diverged.json`) flagged `diverged: true`, excluded from all tables | file contents; `cl2.py:239-248` | — | closed |
| F28 | B7 | PASS | buffer 1000/task stated in every table; buffer filled from task training split (`cl2.py:256-258`) | tables; code | — | closed |
| F29 | B8 | PASS | LoRA oracle-ID footnote on all tables, captions, README | table footnotes | — | closed |
| F30 | B12 | Major | seed-variance note (seeds change permutations) unstated | — | added to `report.md` §4.4 | fixed |
| F31 | J2 | PASS | no secrets/tokens in history | `git log -p` grep | — | closed |
| F32 | J1 | PASS | `.gitignore` covers `data/`, `.venv/`, `__pycache__/` | `.gitignore` | — | closed |
| F33 | J4 | decision | README shows an MIT badge but no LICENSE file exists | repo tree | decision for user (§9) | open |
| F34 | C7 | partial | gate modes have a shared docstring (`cl2.py:86-96`) covering all six; no hard-coded absolute paths (relative `data/`, `outputs/`); seeds set in one place (`set_seed`); no dead code found; defaults documented via argparse help | `cl2.py` | — | closed |
| F35 | A4 | PASS | paired-difference figures ("+4.92 ± 1.20") recomputed: mean ± ddof-0 std of per-seed (method − finetune) ACC on same seed/permulations; definition now stated in `report.md` §4.4 | `outputs/audit/recomputed_numbers.csv` | — | closed |
| F36 | C6 | partial | no GPU available; CPU-only equivalence untestable | hardware | listed as limitation | open |
| F37 | E3 | Major | no multiple-comparisons note | — | added to `report.md` §4.5 | fixed |
| F38 | H1/H5 | Major | no designed poster PDF/PNG exists (poster is text-only `poster_text.md`); not built in this audit | `outputs/phase5/` | left open; text is ready to lay out | open |
| F39 | B3-note | Major | repaired-gate hyperparameters were tuned on Perm5 validation only, then transferred to Perm10 and Split-MNIST without per-benchmark re-tuning | `tune_and_run_repaired.py` (benchmarks loop after single tuning) | recorded as limitation; examiners should be told | open |

## 4. Claims ledger summary

Ledger: `outputs/audit/claims_ledger.csv` — **60 claims** tracked.

- **verified:** 41
- **fix (fixed this audit):** 17
- **unsupported remaining:** 2 (the two open decisions: MIT badge without LICENSE file; Python "3.10+" badge vs 3.9.6 runtime — cosmetic, see §9)

**Ten most important changes:**
1. lr/batch size corrected in three documents (0.05/128, fixed across methods, not tuned).
2. conflict_soft label β=0.1 → β=0.05 (and hybrid γ=0.25 → 0.5 in poster) to match locked configs.
3. magnitude-gate equation rewritten to match code (mean-normalized, clamp max 1).
4. ">80% gated" → measured ~48%.
5. Task-1 retention on Perm10 corrected 14.28±1.57 → 13.44±4.44.
6. EWC-beats-every-gate statement added for Perm5 as well as Perm10.
7. Perm5 `gate:drift` result flagged borderline (0.04-point margin; tie under ddof=1).
8. "statistically outperform" reworded; no significance test claimed; multiple-comparisons note added.
9. References rebuilt: Titans arXiv ID corrected (2501.00663), L2-SP ID removed, 8 missing works added, all verified.
10. Historical prototype-era documents marked SUPERSEDED.

## 5. Verification of headline numbers

All values recomputed by `outputs/audit/recompute.py` from raw JSON; **diff = 0.000 for every cell** (population std, ddof=0, 5 seeds 100–104). Sample table (full table in `outputs/audit/recomputed_numbers.csv`):

| Value as reported | report | recomputed | diff |
|---|---|---|---|
| finetune Perm5 ACC | 88.97 ± 1.85 | 88.97 ± 1.85 | 0.00 |
| finetune Perm10 ACC | 78.21 ± 1.68 | 78.21 ± 1.68 | 0.00 |
| finetune Split ACC | 62.07 ± 1.00 | 62.07 ± 1.00 | 0.00 |
| ewc Perm5 ACC | 93.76 ± 0.36 | 93.76 ± 0.36 | 0.00 |
| ewc Perm10 ACC | 90.22 ± 0.87 | 90.22 ± 0.87 | 0.00 |
| replay Perm10 ACC | 95.22 ± 0.09 | 95.22 ± 0.09 | 0.00 |
| lora Split ACC | 98.73 ± 0.06 | 98.73 ± 0.06 | 0.00 |
| conflict Perm10 ACC | 41.92 ± 1.71 | 41.92 ± 1.71 | 0.00 |
| magnitude Perm10 ACC | 83.13 ± 1.22 | 83.13 ± 1.22 | 0.00 |
| drift Perm5 ACC | 91.89 ± 1.03 | 91.89 ± 1.03 | 0.00 |
| hybrid Perm10 ACC | 83.82 ± 0.54 | 83.82 ± 0.54 | 0.00 |
| conflict_soft Perm5 ACC | 90.68 ± 0.58 | 90.68 ± 0.58 | 0.00 |
| paired diff magnitude vs FT, Perm10 | +4.92 ± 1.20 | +4.92 ± 1.20 | 0.00 |
| diagnostics l1.weight norm conflict / FT | 22.700 / 11.790 | 22.700 / 11.790 | 0.00 |
| drift along d conflict / FT | +2.263 / −0.145 | +2.263 / −0.145 | 0.00 |

One reported number was **wrong and fixed**: `report.md` §5.2 Task-1 retention 14.28±1.57 → 13.44±4.44 (`final_perm10.json` R[-1][0]).

## 6. Statistical notes

- **Std convention:** every table uses population std (numpy `ddof=0`), n=5. Under ddof=1 the intervals widen by ~12%; all tie/difference conclusions below are re-checked under both.
- **Tie rule applied** (mean ± std interval overlap; plus paired-difference sign consistency for "improves"):
  - **Perm5 vs fine-tuning:** EWC, replay, LoRA, gate:conflict → non-overlapping, paired diffs consistent in sign (5/5) → genuine differences. gate:magnitude, layer_cos, hybrid, conflict_soft → overlapping intervals → **ties** (magnitude's paired diffs are +5/5 in sign but intervals overlap: +2.46±1.73). gate:drift → **borderline**: non-overlap by 0.04 points under ddof=0 with +5/5 signs; tie under ddof=1. Treat as a tie unless the convention is stated.
  - **Perm10 vs fine-tuning:** EWC (+12.01±0.86), replay, LoRA, magnitude (+4.92±1.20), drift (+5.01±1.48), hybrid (+5.62±1.21) → non-overlapping with 5/5 positive paired differences → genuine differences. layer_cos, conflict_soft → ties.
  - **Split-MNIST vs fine-tuning:** every buffer-free method ties (61.90–63.96, all overlapping). Replay and LoRA differ hugely (+34.78, +36.66).
- **No significance test is claimed anywhere**; the tie rule is the decision procedure. Six gate variants × three benchmarks = 18 comparisons; some chance findings are expected and this is stated in `report.md` §4.5.

## 7. Code and test results

`tests/test_cl2.py`, run with `.venv/bin/python -m pytest tests/` — **15/15 pass** (fast: 10 in 1.0s; slow: 5 in 4.0s; log `logs/audit_slow_tests.log`).

| test | checks | result |
|---|---|---|
| conflict mask | scales exactly entries with g·d>0 by (1−β·mag) | pass |
| magnitude scale | in [1−β, 1]; equals 1 where d=0 | pass |
| drift condition | blocks g·e<0 (distance expansion) | pass |
| hybrid | multiplies conflict and magnitude factors | pass |
| layer_cos | damps only when cos(g,d)>0 | pass |
| β=0 identity | all four modes leave gradient unchanged | pass |
| decay | d ← decay·d + (p − start) | pass |
| metrics | ACC/BWT/FWT/NEW on hand-computed R matrices | pass |
| data split | 55k/5k/10k disjointness of rows | pass |
| seed sets | {0,1,2} ∩ {100..104} = ∅ | pass |
| determinism | same seed → identical R (CPU) | pass |
| replay buffer | grows by exactly buf per task | pass |
| EWC finiteness | λ=20 run finite, ACC > 50% | pass |
| β=0 ≡ finetune end-to-end | same seed → identical R | pass |
| C4 regression | cl.py 90.8±2.4 vs cl2.py 91.3±1.8 finetune ACC — within 1 std | pass |

**Bugs found by the audit:** none in `cl2.py` itself. Two audit-test expectation bugs were fixed during development (hybrid uses γ independently of β; `d` accumulates relative to task-start weights). Every documentation-vs-code mismatch is listed in §3.

**C1 gate-mode descriptions (code vs report):**
- `conflict` (`cl2.py:117-119`): scales gradient by 1 − β·mag where g·d>0, mag = clamp(|d|/mean|d|, max 1). Report matches (after fixing §3.2 for the shared mag definition).
- `magnitude` (`cl2.py:120-121`): scales by 1 − β·mag regardless of direction. Report matches after fix.
- `layer_cos` (`cl2.py:122-125`): per-tensor cosine c = −⟨g,d⟩/(‖g‖‖d‖); scales by 1 − β·(−c) when c<0, i.e. when cos(g,d)>0. Report matches.
- `drift` (`cl2.py:126-129`): gradient brake on (g·d>0) ∨ (g·e<0), e = θ−θ_start. **Not a loss penalty.** Report describes it as a gradient-level anchor; differences from EWC (Fisher-weighted) and L2-SP (isotropic quadratic loss) are stated. Matches.
- `hybrid` (`cl2.py:130-132`): (1−β·mag·[g·d>0])·(1−γ·mag). Matches (γ=0.5 label fixed).
- `conflict_soft`: conflict with β=0.05, decay 0.9. Label fixed.

## 8. Reproducibility result

Clean-clone smoke test (this audit):
```
git clone /Users/jeet/Desktop/AEGIS /private/var/folders/.../aegis_audit_clone
cd aegis_audit_clone && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python cl2.py --seeds 1 --epochs 1        # README smoke test
```
Result: exit 0, all methods run, EWC divergence guard catches the diverging run at default λ=5000, table produced (~15 s CPU). MNIST downloads via torchvision into `data/` (not committed). Full log: §"clean-clone" below in Changes; raw outputs in the clone. Runtimes: smoke test ≈ 15 s (1 seed, 1 epoch, CPU M4); final 5-seed benchmark ≈ minutes per method (CPU).

## 9. Decisions for the user (Part 7 of the brief)

1. **Project name:** repo/status says **AEGIS**; earlier docs say "Selective Plasticity". STATUS.md Final Verification 6 records the user already chose "AEGIS", with "Selective Plasticity" as the mechanism name. **Confirm** this stands.
2. **Title:** current README title ("Continual Learning with Directional & Magnitude Gradient Gating") no longer overstates the failed conflict variant, but `poster_text.md` still uses "Gradient-Conflict Gating" in its title line. Options: (a) keep as-is; (b) "Selective Plasticity: Displacement-Based Update Damping for Continual Fine-Tuning" (brief's suggestion); (c) "Beyond the Ratchet: Diagnosing and Repairing Displacement-Based Update Gating for Continual Learning". **User must choose**; not changed silently.
3. **License:** README displays an MIT badge but there is no LICENSE file (F33). Either add `LICENSE` (MIT) or remove the badge. Default without a decision: badge removed next session.
4. **Language-model work:** Phase 4 is confirmed next-semester scope (documented as future work in `report.md` §8). No deadline change known; asked only if the user's timeline moved.

## 10. Citation verification

All 14 references verified by web search (subagent report, 2026-11-18):
EWC (arXiv:1612.00796 / DOI 10.1073/pnas.1611835114) · SI (PMLR 70:3987-3995) · MAS (arXiv:1711.09601, ECCV 2018) · LwF (TPAMI 40(12) 2018, DOI 10.1109/TPAMI.2017.2773081; ECCV 2016) · GEM (arXiv:1706.08840) · A-GEM (arXiv:1812.00420, ICLR 2019) · TEM (arXiv:1902.10486) · PCGrad (arXiv:2001.06782, NeurIPS 2020) · L2-SP (**no arXiv record**; cited by title/authors + ICML 2019 CLWor workshop — the common "arXiv:1809.05322" attribution is wrong) · Robins 1995 (DOI 10.1080/09540099550039318) · Goodfellow 2013 (arXiv:1312.6211) · LoRA (arXiv:2106.09685, ICLR 2022) · Titans (**arXiv:2501.00663**, Behrouz, Zhong & Mirrokni — 2412.19837 removed, it is a different paper) · van de Ven & Tolias (arXiv:1904.07734). None removed for being unverifiable; two IDs corrected.

## 11. Changes made / not made

**Changed (this audit):** `outputs/phase5/report.md` (8 fixes), `README.md` (3), `outputs/phase5/poster_text.md` (3), `outputs/phase5/qa.md` (4), `outputs/phase5/captions.md` (3), new `requirements.txt`, new `tests/test_cl2.py`, new `outputs/audit/{recompute.py, recomputed_numbers.csv, claims_ledger.csv, regression/}`, SUPERSEDED banners on `archive/phase0/summary.md` and `archive/phase0/results_explanation.md` (then at `outputs/summary.md` / `outputs/results_explanation.md`), `STATUS.md` final entry. Committed before tagging `sem1-review`.

**Not made, with reasons:**
- No code changes to `cl2.py`/`cl.py` — no confirmed bug in result-producing code (Rule 9).
- No re-run of the 5-seed finals — no fix changed any result (documentation-only fixes).
- No designed poster PDF/slides (H1/H6) — layout work, needs user tooling choice.
- No LICENSE file — user decision (Part 7).
- Title unchanged — user decision (Part 7).

## 12. Remaining work, ranked

1. **(Major, before review)** Lay out `poster_text.md` into a poster PDF/PNG; export both final reports to PDF; assemble `outputs/final/` (done this session where tooling allowed).
2. **(Major, next semester)** Re-tune repaired-gate hyperparameters per benchmark (F39); add SI/MAS/A-GEM/L2-SP baselines; raise seeds to 10.
3. **(Minor)** Resolve the MIT-badge/LICENSE and Python-badge decisions; reconcile `report.md` title with the poster title once the user picks.
4. **(Minor)** ddof=1 vs ddof=0: if the examiner audience prefers sample std, regenerate tables with ddof=1 — conclusions are unchanged except the Perm5 drift borderline case becomes an explicit tie.
