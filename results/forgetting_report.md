# Forgetting / Fade-Curve Probe

**Date**: 2026-06-26 10:30:26  
**Memory model**: (recall-required regime; no LLM)  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (6/6)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | cold first dropped from recall at cycle | INFO | 24 |
| info | cold fully pruned at cycle | INFO | 24 |
| info | cold final | INFO | {'present': 0, 'present_frac': 0.0, 'mean_res': 0.0, 'recall@10': 0.0} |
| info | reinforced final | INFO | {'present': 6, 'present_frac': 1.0, 'mean_res': 123.54, 'recall@10': 1.0} |
| info | pinned final | INFO | {'present': 6, 'present_frac': 1.0, 'mean_res': 3.0, 'recall@10': 1.0} |
| info | revived final | INFO | {'present': 6, 'present_frac': 1.0, 'mean_res': 27.75, 'recall@10': 1.0} |
| hard | COLD resonance declines monotonically while present | PASS |  |
| hard | COLD is eventually forgotten (pruned + dropped from recall) | PASS | {'present': 0, 'present_frac': 0.0, 'mean_res': 0.0, 'recall@10': 0.0} |
| hard | PINNED never fades (present + recall@10==1.0 every cycle) | PASS |  |
| hard | REINFORCED persists (final recall@10==1.0, all present) | PASS | {'present': 6, 'present_frac': 1.0, 'mean_res': 123.54, 'recall@10': 1.0} |
| hard | REVIVED recovers after reinforcement resumes (buried-but-pluckable) | PASS | pre_res=0.51 final_res=27.75 |
| info | contrast: distinctive fact under DEFAULT regime | INFO | res=6.87 tier=long |
| hard | DEFAULT-regime distinctive fact is RETAINED (novelty->long->decay-exempt) | PASS |  |

## Fade milestones

- COLD dropped from recall at cycle **24**, fully pruned at **24** (gradual resonance decay, then a recall cliff at prune).
- REINFORCED + PINNED held recall@10 = 1.0 throughout.
- REVIVED faded then recovered once reinforcement resumed at cycle 20 (buried-but-pluckable).
- Contrast: the SAME kind of distinctive fact is RETAINED under the default regime (novelty->long tier->decay-exempt) - retention is regime/salience dependent.


## Fade curve (per cycle)

Per-cycle mean resonance (present facts) | recall@10:

| cyc | COLD res / r@10 | REIN res / r@10 | PIN res / r@10 | REVV res / r@10 |
|---|---|---|---|---|
| 1 | 2.88 / 1.0 | 7.97 / 1.0 | 3.0 / 1.0 | 3.22 / 1.0 |
| 2 | 2.75 / 1.0 | 11.65 / 1.0 | 3.0 / 1.0 | 3.1 / 1.0 |
| 3 | 2.61 / 1.0 | 15.34 / 1.0 | 3.0 / 1.0 | 2.97 / 1.0 |
| 4 | 2.47 / 1.0 | 19.04 / 1.0 | 3.0 / 1.0 | 2.85 / 1.0 |
| 5 | 2.33 / 1.0 | 22.74 / 1.0 | 3.0 / 1.0 | 2.71 / 1.0 |
| 6 | 2.18 / 1.0 | 26.47 / 1.0 | 3.0 / 1.0 | 2.58 / 1.0 |
| 7 | 2.02 / 1.0 | 30.21 / 1.0 | 3.0 / 1.0 | 2.43 / 1.0 |
| 8 | 1.86 / 1.0 | 33.94 / 1.0 | 3.0 / 1.0 | 2.28 / 1.0 |
| 9 | 1.68 / 1.0 | 37.67 / 1.0 | 3.0 / 1.0 | 2.13 / 1.0 |
| 10 | 1.5 / 1.0 | 41.41 / 1.0 | 3.0 / 1.0 | 1.96 / 1.0 |
| 11 | 1.3 / 1.0 | 45.14 / 1.0 | 3.0 / 1.0 | 1.78 / 1.0 |
| 12 | 1.08 / 1.0 | 48.87 / 1.0 | 3.0 / 1.0 | 1.59 / 1.0 |
| 13 | 0.84 / 1.0 | 52.61 / 1.0 | 3.0 / 1.0 | 1.39 / 1.0 |
| 14 | 0.57 / 1.0 | 56.34 / 1.0 | 3.0 / 1.0 | 1.16 / 1.0 |
| 15 | 0.25 / 1.0 | 60.07 / 1.0 | 3.0 / 1.0 | 0.9 / 1.0 |
| 16 | 0.0 / 1.0 | 63.81 / 1.0 | 3.0 / 1.0 | 0.68 / 1.0 |
| 17 | 0.0 / 1.0 | 67.54 / 1.0 | 3.0 / 1.0 | 0.63 / 1.0 |
| 18 | 0.0 / 1.0 | 71.27 / 1.0 | 3.0 / 1.0 | 0.57 / 1.0 |
| 19 | 0.0 / 1.0 | 75.01 / 1.0 | 3.0 / 1.0 | 0.51 / 1.0 |
| 20 | 0.0 / 1.0 | 78.74 / 1.0 | 3.0 / 1.0 | 2.57 / 1.0 |
| 21 | 0.0 / 1.0 | 82.47 / 1.0 | 3.0 / 1.0 | 4.55 / 1.0 |
| 22 | 0.0 / 1.0 | 86.21 / 1.0 | 3.0 / 1.0 | 6.58 / 1.0 |
| 23 | 0.0 / 1.0 | 89.94 / 1.0 | 3.0 / 1.0 | 8.64 / 1.0 |
| 24 | 0.0 / 0.0 | 93.67 / 1.0 | 3.0 / 1.0 | 10.72 / 1.0 |
| 25 | 0.0 / 0.0 | 97.41 / 1.0 | 3.0 / 1.0 | 12.82 / 1.0 |
| 26 | 0.0 / 0.0 | 101.14 / 1.0 | 3.0 / 1.0 | 14.95 / 1.0 |
| 27 | 0.0 / 0.0 | 104.87 / 1.0 | 3.0 / 1.0 | 17.08 / 1.0 |
| 28 | 0.0 / 0.0 | 108.61 / 1.0 | 3.0 / 1.0 | 19.22 / 1.0 |
| 29 | 0.0 / 0.0 | 112.34 / 1.0 | 3.0 / 1.0 | 21.35 / 1.0 |
| 30 | 0.0 / 0.0 | 116.07 / 1.0 | 3.0 / 1.0 | 23.48 / 1.0 |
| 31 | 0.0 / 0.0 | 119.81 / 1.0 | 3.0 / 1.0 | 25.62 / 1.0 |
| 32 | 0.0 / 0.0 | 123.54 / 1.0 | 3.0 / 1.0 | 27.75 / 1.0 |

