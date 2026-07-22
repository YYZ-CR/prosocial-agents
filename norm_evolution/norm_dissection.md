# Norm dissection (item 3)

- Distinct norm texts: **49** (the 1 seed text + 48 optimized).
- Baseline arm: all 30 baseline slots are the unchanged seed norm (`baseline_all_seed=True`).
- **State-contingent test:** 42 of 48 optimized norms (88%) condition catch on the *measured* stock level -- the 'state-contingent' characterization is real, not just 'longer'.
- Optimized norms carry explicit numeric caps in 32/48, a moratorium clause in 33/48, an amendment rule in 44/48, and enforcement/logging language in 26/48.
- Mean optimized norm: 1178 chars, 6.6 clauses (vs the 130-char, 2-clause seed).

## Highest-scoring norm (opus48, textgrad, iteration 3, reward 2.500)

> 1. Before fishing each month, measure the current stock S.
> 2. If S ≥ 100 (lake at full capacity): each fisher may catch up to 10 fish (total cap 50), preserving the sustainable yield.
> 3. If 50 < S < 100 (lake recovering): each fisher may catch up to (S-50)/20 fish (total cap (S-50)/4), ensuring at least 3/4 of the stock above 50 remains to regenerate.
> 4. If S ≤ 50 (lake depleted): no fishing is allowed that month, so the stock can rebuild.
> 5. Catches must be equal across all fishers in every month; no one takes another's unused share.
> 6. Any change to these rules requires agreement from at least 4 of the 5 fishers.
> 7. Fishers should recognize that following this graduated rule keeps the fishery productive and avoids the losses of shutdowns or collapse, making it in everyone's interest to keep it rather than renegotiate.

Features: 7 clauses, 3 stock-conditional, 2 numeric caps, moratorium=True, amendment=True.
