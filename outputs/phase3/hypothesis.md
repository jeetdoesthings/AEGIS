# Ratchet Hypothesis Test: Mechanism Evaluation

**Decision**: Ratchet hypothesis is **SUPPORTED** by this diagnostic.

## Empirical Evidence (`outputs/phase3/diag_all.csv`):
- **Weight L2 Norm (Task 5, l1.weight)**: gate:conflict = 22.700, fine-tuning = 11.790, gate:magnitude = 12.291.
- **Drift along d (Task 5, l1.weight)**: gate:conflict = 2.263, fine-tuning = -0.145, gate:magnitude = -0.016.

## Diagnostic Analysis:
The conflict gate exhibits strictly increasing and larger weight norms and drift along d than both fine-tuning and the magnitude gate, indicating that blocking reversals causes directional runaway along d.
