# Precision Under Load (#1 trust)

**Date**: 2026-06-27 01:10:56  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (2/2)  
**Soft/LLM warnings**: 1 (relevance gap is positive for every topic (relevant out-scores distractors))  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | distractors added | INFO | 150 of 150 |
| info | total rows in store | INFO | 180 |
| info | zephyrine-migration | INFO | prec@5=5/5 top1=rel |
| info | tanager-billing | INFO | prec@5=4/5 top1=rel |
| info | kestrel-compliance | INFO | prec@5=5/5 top1=rel |
| info | borealis-support | INFO | prec@5=4/5 top1=rel |
| info | orrery-contract | INFO | prec@5=4/5 top1=rel |
| info | vantyx-procurement | INFO | prec@5=5/5 top1=rel |
| info | mean precision@5 | INFO | 0.9 |
| info | top-1 relevant | INFO | 6/6 |
| info | clean-cut (top-R == relevant set) | INFO | 3/6 |
| info | relevance gaps (min_relevant - max_distractor) | INFO | [0.238, -0.023, 0.147, -0.046, 0.023, 0.103] |
| hard | top-1 is relevant for EVERY topic query (no distractor wins the #1 slot) | PASS | 6/6 |
| hard | mean precision@5 >= 0.8 under load | PASS | 0.9 |
| soft | relevance gap is positive for every topic (relevant out-scores distractors) | WARN | [0.238, -0.023, 0.147, -0.046, 0.023, 0.103] |
| info | adaptive gate: relevant kept / relevant dropped / distractors dropped | INFO | 27 / 3 / 89 |
| soft | adaptive gate drops distractors without nuking relevant facts | PASS | dropped_dis=89 dropped_rel=3 |
