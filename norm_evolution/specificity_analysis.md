# Does specificity predict outcome? (item 4)

Specificity index = z(clauses) + z(numeric caps) + z(stock-conditionals) + z(log length) + z(institutional flags). Higher = more fully specified; the vague seed sits at the low end.

## Correlation of reward with specificity

| Group | n norms | Pearson r | Spearman rho | OLS slope (reward per +1 index) |
|---|---|---|---|---|
| Gemma 4 31B | 15 | +0.35 | +0.51 | +0.030 |
| GPT-5.6 Luna | 15 | -0.20 | -0.20 | -0.035 |
| Sonnet 5 (no reas.) | 15 | +0.45 | +0.27 | +0.016 |
| Sonnet 5 (reas.) | 15 | +0.54 | +0.61 | +0.016 |
| Opus 4.8 | 15 | -0.36 | -0.20 | -0.006 |
| GPT-5.6 Sol | 15 | -0.30 | -0.28 | -0.031 |
| **Pooled, non-collapse (5 models)** | 75 | -0.03 | +0.16 | -0.004 |
| **Pooled, all six models** | 90 | -0.03 | +0.10 | -0.006 |

## Robustness: alternative index definitions (pooled, non-collapse)

| Index definition | Pearson r | Spearman rho |
|---|---|---|
| Full (5 components) | -0.03 | +0.16 |
| Length only | -0.06 | +0.16 |
| Clause count only | -0.01 | +0.22 |
| Numeric caps + stock-conditionals | +0.00 | +0.28 |

**Headline:** across the five non-collapse models, reward is essentially flat in specificity (Pearson r $= -0.03$, $R^2 = 0\%$; Spearman $\rho = +0.16$): a fully specified institution scores no better than the vague one-sentence seed. Per-model OLS slopes are all below $0.035$ reward points per index unit, so moving a norm across the entire observed specificity range shifts predicted reward by less than the fixed-norm noise floor. The per-model correlations are small and flip sign across models ($-0.36$ to $+0.54$)---the signature of noise, not a systematic specificity effect. The weak positive Spearman does not survive as a Pearson association and is not robust across index definitions.
