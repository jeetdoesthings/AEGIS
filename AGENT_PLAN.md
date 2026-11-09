# AGENT_PLAN.md: Selective Plasticity, finish the project

You are a coding and research agent. Read this whole file, then work through the phases in order. Do not skip ahead. After every task, update `STATUS.md` (format in section 2) before starting the next one.

Total target: take the project from a half-finished prototype to a complete, defensible final submission.

---

## 0. Context (read first)

**Topic.** Catastrophic forgetting: a neural network loses old skills when trained on new ones. The project tests one idea for reducing it, called the gradient-conflict gate: store how learning each old task moved every weight (`d`), then shrink new updates that would push a weight back against `d`. No old data is stored.

**Goal.** A fair, reproducible comparison of the gate against standard methods (fine-tuning, EWC, replay, LoRA), with honest numbers on forgetting and new-task learning, an explanation of why the gate behaves as it does, and a first test on a small language model.

**Existing files.**
- `cl.py`: working prototype (Permuted-MNIST, MLP 784-256-256-10, plain SGD, methods `finetune`, `ewc`, `replay`, `lora`, `gate:<mode>:<beta>` with modes `conflict`, `magnitude`, `layer_cos`). Metrics: ACC, BWT, FWT, NEW. **Never edit `cl.py`.** It is the frozen reference. Make changes in `cl2.py` (a copy).
- `outputs/`: results from the first runs, `results_<tag>.json`.

**Metric definitions** (R[i][j] = accuracy on task j after training through task i, T tasks):
- ACC = mean_j R[T-1][j]
- BWT = mean over j<T-1 of (R[T-1][j] - R[j][j]); negative means forgetting
- FWT = mean over j>=1 of (R[j-1][j] - untrained accuracy on task j)
- NEW = mean_j R[j][j]; accuracy right after learning each task. Always report it, because a method can look good on BWT by refusing to learn.

**Results so far** (Permuted-MNIST, 5 tasks, 3 seeds, test set, mean ± std in %):

| Method | ACC | BWT | NEW |
|---|---|---|---|
| finetune | 91.57 ± 1.49 | -6.77 ± 1.96 | 96.99 ± 0.09 |
| replay (200/task) | 95.19 ± 0.09 | -2.22 ± 0.07 | 96.96 ± 0.04 |
| lora (uses task ID at test) | 93.79 ± 0.12 | 0.00 ± 0.00 | 93.79 ± 0.12 |
| ewc | 9.80 ± 0.00 | -21.59 ± 0.01 | 27.07 ± 0.01 |
| gate:conflict:1.0 | 15.89 ± 0.41 | -51.69 ± 12.17 | 57.24 ± 9.92 |
| gate:conflict:0.5 | 44.93 ± 2.48 | -59.29 ± 3.04 | 92.36 ± 0.21 |
| gate:layer_cos:1.0 | 91.76 ± 1.43 | -6.51 ± 1.89 | 96.97 ± 0.10 |
| gate:magnitude:1.0 | 92.36 ± 0.32 | -3.65 ± 0.39 | 95.28 ± 0.06 |

**Known problems you must fix or account for:**
1. **EWC is broken.** It scored exactly 9.80% at lambda = 1000, 5000 and 10000, identical to every digit. 9.80% is the share of class 0 in the MNIST test set, which is what argmax returns when outputs are NaN. Suspected cause: lambda too large for lr 0.05 (the quadratic penalty makes SGD diverge). Treat all EWC numbers and every "beats EWC" claim as invalid until repaired.
2. **The conflict gate hurts.** `gate:conflict` is worse than plain fine-tuning on both forgetting and new-task learning. Hypothesis (untested): the gate only blocks reversals, so weights ratchet further along `d` and grow without bound.
3. **`gate:magnitude` is a tie with fine-tuning on ACC** (92.36 ± 0.32 vs 91.57 ± 1.49, within one std). Its BWT is lower and task 1 held up better (91.0 vs 81.1), but 3 seeds only suggest this.
4. **Tuning used the test set.** Hyperparameters must be chosen on a held-out validation split.
5. **Little headroom.** Fine-tuning already scores 91.6%, so methods barely separate. Use harder settings (10 tasks, second benchmark).

---

## 1. Rules (apply to everything)

1. **No invented numbers.** Every number in any document must come from a result file you produced. Cite the file.
2. **No cherry-picking.** Keep every run, including failed and negative ones. Never delete results. Failed runs go in `outputs/failed/` with a note on why.
3. **Tune on validation only.** The test set is touched only for final reported runs, with hyperparameters locked beforehand.
4. **Equal tuning effort.** Every method gets the same number of tuned configurations (at most 6) on the validation split.
5. **Ties.** Two methods are tied if their mean ± std intervals overlap. Also report the per-seed paired difference versus fine-tuning (same seed means same permutations). Never write "beats" for a tie.
6. **Label every claim** in written text as MEASURED (read from a result file) or HYPOTHESIS (a reasoned guess). Do not use "prove" or "proves" for experimental evidence; use "suggests".
7. **Negative results are valid.** If the gate does not work, say so plainly. Do not tune until it wins by chance.
8. **LoRA caveat.** LoRA uses task ID at test time. Always footnote this.
9. **Run long jobs in the background** with `nohup`, logging to `logs/<tag>.log`; poll the log. On macOS prefix with `caffeinate -i`. Smoke-test every new script with a tiny config (`--seeds 1 --epochs 1`) before the real run, and time it to estimate the real run.
10. **Reproducibility.** Every run writes a config JSON (all arguments, git commit hash, torch version, device, seed). Commit to git after every task.
11. **Ask the user only when blocked** (missing access, a decision that changes scope). Otherwise make the conservative choice, record it in `STATUS.md`, and continue.

---

## 2. STATUS.md format

Create `STATUS.md` at the start. After every task append:

```
## Task <id> <name>
- status: done | partial | blocked
- what changed: <files>
- key numbers: <only from result files, with file path>
- problems / decisions: <anything the user must know>
- next: <next task id>
```

---

## 3. Folder layout

```
project/
  cl.py                  # frozen, never edit
  cl2.py                 # working copy, all changes here
  lm/                    # phase 4 code
  outputs/
    phase1/ phase2/ phase3/ phase4/ phase5/ failed/
  logs/
  configs/
  STATUS.md
  AGENT_PLAN.md
```

---

## Phase 0: Setup (30 min)

**Task 0.1: Environment.** Create `.venv`, install `torch torchvision matplotlib numpy pandas`. Print torch version and which device is available (cuda, mps, cpu). Use CPU on Mac unless MPS is confirmed to work; test MPS with the smoke test and fall back to CPU on any error.

**Task 0.2: Freeze.** `git init`, commit `cl.py` unchanged, copy it to `cl2.py`. Create the folders and `STATUS.md`.

**Acceptance:** `python cl2.py --seeds 1 --epochs 1` runs and prints a table. Record the wall-clock time; multiply by the number of seeds, epochs and methods to estimate later runs.

---

## Phase 1: Fix the baselines and make the evaluation honest (target: half a day)

### Task 1.1: Validation split and seed control
Edit `cl2.py`:
- Split MNIST train into 55,000 train / 5,000 validation (fixed split seed, same for all methods; stratified is optional).
- Add `--eval_split val|test`. Training never sees validation. All tuning runs use `val`; final runs use `test`.
- Add `--seed_start N` so tuning seeds (0,1,2) and final seeds (100-104) never overlap. Seeds control the task permutations, so different seeds mean different task sequences.
- Add `--out TAG` so each run writes `outputs/<phase>/results_<TAG>.json` and a config file directly, instead of overwriting `results.json`.

**Acceptance:** a run with `--eval_split val` reports numbers that differ from `--eval_split test`; no file is overwritten between runs.

### Task 1.2: Repair EWC
1. Reproduce the failure: run EWC at lambda = 5000 for task 2 only, print the loss every 50 steps and check `torch.isfinite(loss)`. Confirm or reject the NaN diagnosis. Save the printout in `outputs/phase1/ewc_diagnosis.txt`.
2. Add a divergence guard to the training loop: if the loss is not finite, stop that run, mark it `diverged: true` in the JSON, and continue with the next run. Never report a diverged run's accuracy as a result.
3. Sweep lambda in {1, 5, 20, 100} on validation, 3 tuning seeds. If any value still diverges, add `--clip` (gradient-norm clipping at 5.0) and rerun that value.
4. Report which lambda is best on validation (by ACC; break ties with BWT, then NEW). Write one paragraph explaining the cause, labeled HYPOTHESIS if it was not directly verified.

**Acceptance:** at least one lambda gives finite training and a final ACC well above 10%.

### Task 1.3: Fair tuning of everything
Tune each method on validation with 3 tuning seeds and at most 6 configurations:

| Method | Grid |
|---|---|
| finetune | none (1 config) |
| ewc | lambda in {1, 5, 20, 100} (+ clipping if needed) |
| replay | buffer per task in {50, 200, 1000} |
| lora | rank in {4, 8, 16} |
| gate:conflict | beta in {0.25, 0.5, 1.0}, decay in {1.0, 0.9} |
| gate:magnitude | beta in {0.25, 0.5, 1.0}, decay in {1.0, 0.9} |
| gate:layer_cos | beta in {0.5, 1.0} |

Save a table `outputs/phase1/tuning_val.csv` with every configuration and its metrics. Lock the best configuration per method (by ACC, ties by BWT, then NEW) into `configs/locked_perm5.json`.

### Task 1.4: Final run on Permuted-MNIST, 5 tasks
Run every method with its locked configuration on the test split, 5 final seeds (100-104). Save `outputs/phase1/final_perm5.json`, the table (CSV and image), and the forgetting curve (accuracy on task 1 versus tasks learned, mean ± std band). Also produce accuracy-matrix heatmaps per method.

**Acceptance:** all methods finite, 5 seeds each, table with mean ± std, paired difference versus fine-tuning per method.

---

## Phase 2: Harder benchmarks (target: half a day)

### Task 2.1: 10-task Permuted-MNIST
Run all methods with `--tasks 10`. Retune only if a locked config diverges (record this). Final seeds 100-104, test split. Output `outputs/phase2/final_perm10.*`.

### Task 2.2: Split-MNIST, domain-incremental
Add `--bench splitmnist` to `cl2.py`. Task k contains digits (2k, 2k+1) with label = digit - 2k, so the model uses one shared 2-way output head and no task ID (domain-incremental). Five tasks. Retune each method on validation with the same grid budget, lock configs in `configs/locked_split.json`, then final test runs with 5 seeds. Output `outputs/phase2/final_split.*`.
LoRA again uses task ID at test time; footnote it.

**Acceptance:** two additional benchmark tables with the same format as Phase 1. Write 5 sentences comparing which methods change rank across benchmarks (MEASURED only).

---

## Phase 3: Diagnose the conflict gate and test a repair (target: 1-2 days)

This phase is the core research contribution. Be rigorous.

### Task 3.1: Diagnostics logging
Add a `--diag` flag that, for each task and each parameter tensor, logs (averaged over steps, written to `outputs/phase3/diag_<TAG>.csv`):
- L2 norm of the weights
- fraction of gradient entries that were gated (scaled by less than 1)
- mean scale factor applied
- projection of (current weights minus weights at the start of this task) onto `d`, normalized by the norm of `d` (the "drift along d")
- cosine between the proposed step and `d`

Run `diag` for `finetune`, `gate:conflict` (best tuned config) and `gate:magnitude` (best tuned config), 3 seeds, Permuted-MNIST 5 tasks.

### Task 3.2: Test the ratchet hypothesis
Plot weight norm and drift along `d` per layer across tasks for the three methods. Decision rule, written into `outputs/phase3/hypothesis.md`:
- If the conflict gate shows clearly larger and growing weight norm and drift along `d` than fine-tuning and the magnitude gate, label the ratchet hypothesis SUPPORTED (by this diagnostic).
- If not, label it NOT SUPPORTED, and inspect two alternatives using the same logs: (a) gating makes steps so sparse that new tasks are barely learned (check the gated fraction and NEW), (b) the quantity `d` is dominated by task 1's movement from random initialization, so it carries little task-specific information. Report what the logs show.
Do not claim a cause that the logs do not show.

### Task 3.3: Implement repaired variants
Add these modes to `Gate`. Each must be a pure function of the current gradient, `d`, and the start-of-task or end-of-previous-task weights `theta_star`:
- `drift`: for weights with high `mag`, also scale down steps that increase the distance from `theta_star` (the anti-ratchet term). Step = -g, deviation e = theta - theta_star; the step increases distance when g*e < 0. Scale = 1 - beta*mag*[g*d > 0 or g*e < 0].
- `hybrid`: multiply the conflict factor and the magnitude factor: (1 - beta*mag*conflict) * (1 - gamma*mag), with `gamma` in {0.25, 0.5}.
- `conflict_soft`: the conflict gate with beta in {0.05, 0.1}, to test whether it is simply too strong.

Keep names and definitions in a docstring. Tune on validation (budget: 6 configs per mode), lock, run final on the test split, 5 seeds, on Permuted-MNIST 5 and 10 tasks and Split-MNIST.

### Task 3.4: Layer analysis
For the best gate variant, plot per-layer average gated fraction by task (heatmap: rows = parameter tensors, columns = tasks). State in MEASURED language which layers are gated most.

### Task 3.5: Write the finding
`outputs/phase3/findings.md`, max 400 words. Structure: what was tested, what the numbers show (MEASURED), what the diagnostics show, what is still unexplained, and whether any gate variant is distinguishable from fine-tuning (use the tie rule). If no variant is, say that clearly.

**Acceptance:** diagnostics CSVs and plots exist; the repaired variants are in the final tables; `findings.md` has no claim without a file reference.

---

## Phase 4: Language-model experiment (target: 2-4 days; riskiest phase)

**Go/no-go gate.** Start Phase 4 only if Phases 1-3 are complete and the user confirms at least 3 days remain. If not, skip to Phase 5 and describe Phase 4 as future work. Run Phase 4 on a GPU (Colab or similar); do not run it on CPU.

### Task 4.1: Setup
- Model: GPT-2 small (124M) from Hugging Face `transformers`, pretrained weights.
- Domains (4, in this order): general encyclopedic text (WikiText-103), source code, biomedical abstracts, news. Use Hugging Face `datasets`. Verify each dataset loads; if a dataset is unavailable, substitute a similar one and record it in `STATUS.md`.
- Per domain: tokenize, pack into sequences of 256 tokens, 2,000 training steps at batch size 8, 200 held-out sequences for evaluation.
- Metric: perplexity per domain on its held-out set, evaluated after training on each domain. Forgetting = increase in perplexity on earlier domains. Report lower-is-better and say so on every table and plot.

### Task 4.2: Optimizer handling (important)
Adam divides updates by a running gradient scale, which cancels per-weight gradient shrinking. Therefore gate the final update, not the gradient. Implement `GatedAdam` that computes the Adam update `upd = m_hat / (sqrt(v_hat) + eps)`, applies the gate to `upd` (the step is `-upd`; a conflict is `upd * d > 0`), then applies `p -= lr * upd_gated`. Reference sketch (verify by running):

```python
class GatedAdam(torch.optim.Optimizer):
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
                    upd = self.gate.scale_update(p, upd)  # per-element scale in [0, 1]
                p.add_(upd, alpha=-group["lr"])
```

The gate class must keep `d` per parameter (dictionary keyed by parameter). Memory cost is two extra copies of the weights (`d` and the start-of-task weights); confirm it fits on the GPU.

### Task 4.3: Methods
Run with 1 seed first, then 3 seeds if time allows:
- fine-tuning (AdamW, same lr)
- replay (500 stored sequences per domain, mixed into every batch)
- EWC (diagonal Fisher from 200 batches, lambda tuned on validation in a small grid; use a divergence guard)
- LoRA with one adapter per domain (`peft`), domain ID given at test time (footnote as before)
- gate:magnitude and the best conflict-family variant from Phase 3 (apply to attention and MLP weights; leave embeddings ungated unless a reason is found)
Tune learning rate once for fine-tuning on validation and reuse it for all methods; give each other method at most 3 configurations.

### Task 4.4: Output
Perplexity matrix per method (rows = after domain i, columns = domain j), a forgetting plot, and a summary table. Same tie and labeling rules. If the gate does not help, report that.

**Acceptance:** at least fine-tuning, replay, one gate variant, run on 4 domains, 1 seed, with tables and plots; or a written note in `STATUS.md` explaining exactly where it stopped and why.

---

## Phase 5: Final deliverables (start in parallel from Phase 1; finish last)

### Task 5.1: Results pack
`outputs/phase5/`: final tables (CSV plus images) for every benchmark, forgetting curves, heatmaps, the diagnostics plots, the language-model results if Phase 4 ran. Every figure has a caption stating the benchmark, number of seeds, metric and "mean ± std".

### Task 5.2: Report draft
`outputs/phase5/report.md` with sections: 1 Introduction (the problem, why RAG and long context do not address it), 2 Related work (EWC, replay, GEM, LoRA, Synaptic Intelligence, Titans), 3 Method (the gate, its variants, the equations), 4 Experimental setup (datasets, model, optimizer, seeds, validation protocol, metrics), 5 Results (tables and figures, MEASURED language), 6 Analysis (diagnostics, hypotheses clearly labeled), 7 Limitations, 8 Future work. Related work: do not claim the gate is novel. State that it is related to Synaptic Intelligence (tracks weight movement) and to GEM (gradient conflict), and describe what the experiments test.

### Task 5.3: Claim audit
Search the report and all written text for: "prove", "solve", "solved", "beats", "outperforms", "novel", "state-of-the-art". For each hit, confirm a result file supports it under the tie rule, or rewrite it. Write `outputs/phase5/claim_audit.md` listing each hit and its resolution.

### Task 5.4: Presentation material
- `outputs/phase5/poster_text.md`: problem, setup, results (final table), main finding, limitations, next steps. Plain language.
- `outputs/phase5/qa.md`: the 15 most likely examiner questions with short answers using real numbers (dataset and where it comes from, model and why it is small, how forgetting is measured, why validation tuning, what the negative result means, why EWC needed repair, LoRA caveat, what is new, what is next). Flag answers where the results do not support a strong claim.
- `outputs/phase5/script_60s.md`: a 60-second spoken script using real numbers.

### Task 5.5: Final check
Run the whole pipeline from a clean clone on the smoke-test config to confirm reproducibility. Write `README.md` with the exact commands that regenerate each table and figure.

---

## Final deliverables checklist

- [ ] `cl2.py` with validation split, seed control, divergence guard, diagnostics, repaired variants
- [ ] Repaired EWC with a stated lambda and a no-NaN confirmation
- [ ] Tuning tables (validation) and locked configs
- [ ] Final test tables: Permuted-MNIST 5 tasks, 10 tasks, Split-MNIST (5 seeds each)
- [ ] Diagnostics, hypothesis decision, findings
- [ ] Language-model results, or a documented stop point
- [ ] Report draft, claim audit, poster text, Q&A, 60-second script
- [ ] `README.md`, `STATUS.md` complete

## If time runs short, cut in this order

1. Phase 4 (language model): present as future work.
2. Split-MNIST (keep 10-task Permuted-MNIST).
3. Seeds from 5 to 3.
4. Layer heatmap (Task 3.4).

Never cut: EWC repair, validation split, the claim audit, the report.
