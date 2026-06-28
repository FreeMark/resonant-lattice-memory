# Scale Ceiling (recall + latency at 50k live rows)

**Date**: 2026-06-27 17:57:29  
**Memory model**: (no LLM)  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (3/3)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | golden needles planted | INFO | 30 |
| info | final live rows | INFO | 48052 |
| info | final DB size MB | INFO | 578.0 |
| info | recall@10 trajectory | INFO | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| info | latency_ms trajectory | INFO | [48.2, 64.0, 81.0, 89.6, 101.0, 115.8, 120.5, 133.3, 146.4, 149.1] |
| hard | recall@10 stays >=0.95 at 48052 live rows | PASS | 1.0 |
| hard | recall@1 stays >=0.90 at full scale | PASS | 1.0 |
| hard | recall@10 did not degrade vs the first checkpoint | PASS | 1.0 -> 1.0 |
| info | latency growth (first -> last) | INFO | 48.2ms -> 149.1ms |
