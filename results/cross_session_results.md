# Cross-Session Business Memory (T4)

**Date**: 2026-06-25 14:26:43  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (4/4)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | acme facts visible in session 2 | INFO | 3 |
| hard | session-1 spend (4050/$40.50) recalled in session 2 via entity recall | PASS |  |
| hard | session-2 also sees the new 5250 spend | PASS |  |
| info | relational_recall(acme corp, located_in, ?) | INFO | [('boston', 'graph')] |
| hard | planted relation (acme corp, located_in, boston) persists across sessions | PASS |  |
| info | tool episodes visible in session 2 | INFO | 2 |
| hard | tool episodes from BOTH sessions persist | PASS | 2 episodes |
| info | narrative entries in session 2 | INFO | 1 |
| info | narrative text | INFO | The user requested onboarding Acme Corp and approving its initial $40.50 monthly expenditure, and the assistant recorded the company’s details and granted approval for the spend using the --request-ap |
| soft | session-1 narrative persisted and mentions Acme | PASS | 1 entr(y/ies) |
