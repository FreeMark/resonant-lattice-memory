# Cross-Entity Contamination (#2 trust)

**Date**: 2026-06-27 01:10:47  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (4/4)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | collision-pair fact ids | INFO | {'Acme Corp': 1, 'Acme Inc': 2, 'Globex LLC': 3, 'Globex Holdings': 4, 'Initech': 5, 'Initrode': 6} |
| hard | all collision-pair facts stored as DISTINCT rows (no entity-merge corruption) | PASS | merged=[] distinct_ids=6/6 |
| info | T2 direct: wrong attributions | INFO | none |
| hard | T2 direct: every query returns the RIGHT entity + RIGHT amount (content-verified) | PASS | 0/6 wrong |
| info | distractors loaded | INFO | 150 added of 150 (rest merged) |
| info | total rows | INFO | 156 |
| info | T3 under load: wrong attributions | INFO | none |
| hard | T3 under load: every query returns the RIGHT entity + RIGHT amount (content-verified) | PASS | 0/6 wrong |
| info | T4 shared-value wrong attributions | INFO | none |
| hard | T4: same-amount entities are disambiguated (query returns the RIGHT entity) | PASS | 0/3 wrong |
