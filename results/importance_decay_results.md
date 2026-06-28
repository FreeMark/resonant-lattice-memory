# Importance-Weighted Decay (importance != frequency)

**Date**: 2026-06-27 16:34:49  
**Memory model**: (no LLM)  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (4/4)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | feature OFF — important retained / generic retained | INFO | 0/3  0/3 |
| hard | control: with the feature OFF, unused important facts fade like generic ones | PASS | imp=0 gen=0 |
| info | feature ON (0.6) — important retained / generic retained | INFO | 3/3  0/3 |
| hard | with the feature ON, UNUSED important facts are RETAINED (importance != frequency) | PASS | 3/3 |
| hard | the feature is SELECTIVE — generic noise still fades (no unbounded retention) | PASS | generic retained=0 |
| hard | ON retains strictly more important facts than OFF (the feature is the cause) | PASS | off=0 on=3 |
