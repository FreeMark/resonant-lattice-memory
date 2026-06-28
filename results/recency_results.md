# Supersession / Recency (#3 trust)

**Date**: 2026-06-27 01:13:20  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (4/4)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | update merged into old row (new value at risk) | INFO | 0/30 |
| info | new value present in top-5 | INFO | 30/30 |
| info | CURRENT value is top-1 | INFO | 18/30 |
| info | STALE value is top-1 (failure) | INFO | 12/30 |
| info | stale-top examples | INFO | [("What are Tanager Corp's payment terms?", "got stale: 'Tanager Corp payment terms are Net-30.'"), ('Where is Vantyx Corp headquartered?', "got stale: 'Vantyx Corp is headquartered in Boise.'"), ("What is Borealis Corp's contract status?", 'got stale: "Borealis Corp\'s contract status is trial."'), |
| hard | value-update never silently dropped (current value recallable in top-5) for every update | PASS | 30/30 |
| hard | no update was swallowed as a reinforcement of the stale value | PASS | 0 merged |
| soft | autonomous recall ranks the CURRENT value top-1 (characterized, not a guarantee) | PASS | current_top=18/30, stale_top=12 |
| hard | conflict->resolve supersedes the stale fact (retired as history, winner freed) | PASS |  |
| hard | superseded (stale) fact is withheld from normal recall | PASS |  |
