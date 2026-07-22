# Component-metric re-analysis of the norm-optimization grid

Each optimizer effect is the mean shift over the fixed-norm baseline (`delta_mean`), tested against that cell's baseline across-iteration noise floor (`noise_spread`). `beats_noise` requires a positive shift larger than the noise floor.

## Total gain R  (higher better, bound=600)

| Model | Baseline mean (best) | Headroom | Noise floor (spread / std) | TextGrad Δ | ACE Δ | Beats noise? |
|---|---|---|---|---|---|---|
| Gemma 4 31B | 452 (538) | 62.00 | 135.00 / 47.91 | +75.20 | +74.30 | no |
| GPT-5.6 Luna | 394 (553) | 47.00 | 248.00 / 89.48 | +32.80 | -125.00 | no |
| Sonnet 5 (no reas.) | 414 (546) | 54.50 | 196.00 / 67.86 | +4.60 | +26.90 | no |
| Sonnet 5 (reas.) | 503 (562) | 38.00 | 132.00 / 44.88 | -0.50 | -6.30 | no |
| Opus 4.8 | 556 (590) | 10.00 | 95.00 / 33.72 | +24.60 | -47.80 | no |
| GPT-5.6 Sol | 115 (125) | 475.00 | 25.00 / 12.25 | +59.00 | -15.00 | textgrad |

## Survival months m  (higher better, bound=12)

| Model | Baseline mean (best) | Headroom | Noise floor (spread / std) | TextGrad Δ | ACE Δ | Beats noise? |
|---|---|---|---|---|---|---|
| Gemma 4 31B | 12 (12) | 0.00 | 2.50 / 1.00 | -0.40 | +0.50 | no |
| GPT-5.6 Luna | 10 (12) | 0.00 | 5.50 / 2.37 | -0.10 | -3.30 | no |
| Sonnet 5 (no reas.) | 12 (12) | 0.00 | 0.00 / 0.00 | +0.00 | +0.00 | no |
| Sonnet 5 (reas.) | 12 (12) | 0.00 | 0.00 / 0.00 | +0.00 | +0.00 | no |
| Opus 4.8 | 12 (12) | 0.00 | 0.00 / 0.00 | +0.00 | +0.00 | no |
| GPT-5.6 Sol | 1 (2) | 10.50 | 0.50 / 0.24 | +1.30 | -0.30 | textgrad |

## Over-usage o  (lower better, bound=0)

| Model | Baseline mean (best) | Headroom | Noise floor (spread / std) | TextGrad Δ | ACE Δ | Beats noise? |
|---|---|---|---|---|---|---|
| Gemma 4 31B | 0 (0) | -0.00 | 0.33 / 0.12 | +0.07 | +0.02 | no |
| GPT-5.6 Luna | 0 (0) | -0.00 | 0.50 / 0.20 | -0.11 | -0.31 | no |
| Sonnet 5 (no reas.) | 0 (0) | -0.00 | 0.25 / 0.09 | -0.07 | +0.09 | no |
| Sonnet 5 (reas.) | 0 (0) | 0.08 | 0.17 / 0.07 | +0.01 | +0.10 | no |
| Opus 4.8 | 0 (0) | -0.00 | 0.00 / 0.00 | -0.00 | -0.00 | no |
| GPT-5.6 Sol | 1 (1) | 0.75 | 0.25 / 0.12 | +0.08 | -0.15 | no |

## Equality e  (higher better, bound=1)

| Model | Baseline mean (best) | Headroom | Noise floor (spread / std) | TextGrad Δ | ACE Δ | Beats noise? |
|---|---|---|---|---|---|---|
| Gemma 4 31B | 1 (1) | 0.05 | 0.16 / 0.05 | +0.05 | +0.08 | no |
| GPT-5.6 Luna | 1 (1) | 0.01 | 0.20 / 0.07 | +0.01 | -0.09 | no |
| Sonnet 5 (no reas.) | 1 (1) | 0.01 | 0.04 / 0.02 | -0.05 | +0.00 | no |
| Sonnet 5 (reas.) | 1 (1) | 0.03 | 0.05 / 0.02 | +0.02 | +0.03 | no |
| Opus 4.8 | 1 (1) | 0.01 | 0.02 / 0.01 | +0.01 | +0.00 | no |
| GPT-5.6 Sol | 1 (1) | 0.33 | 0.02 / 0.01 | +0.03 | +0.04 | textgrad, ace |

**Headline:** across the three prof-named metrics (R, m, o) x 6 models x 2 optimizers = 36 tests, an optimizer cleared its own noise floor in **2** — and in **0** once GPT-5.6 Sol is excluded. Both Sol hits are TextGrad, whose apparent gain is the iteration-0 (un-optimized seed) norm catching one lucky episode before collapsing back; it is not a learned improvement. For every model that does not collapse, survival $m$ and over-usage $o$ are already at ceiling/floor (no headroom), and total gain $R$ moves less than the fixed-norm baseline's own across-iteration noise.
