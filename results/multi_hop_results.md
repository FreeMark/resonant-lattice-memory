# Multi-Hop Inference + Conflict Machinery

**Date**: 2026-06-25 14:26:30  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (8/8)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | acme relations | INFO | [('acme corp', 'located_in', 'boston')] |
| hard | deterministic triple (acme corp, located_in, boston) extracted | PASS |  |
| info | inferred from acme corp | INFO | {'massachusetts': 2, 'usa': 3} |
| hard | inference reaches boston (1 hop is direct; >=2 via chain) | PASS |  |
| hard | transitive inference reaches massachusetts (2 hops) | PASS | hops=2 |
| hard | transitive inference reaches usa (3 hops) | PASS | hops=3 |
| hard | inference wrote NOTHING (no-write invariant) | PASS | fact_relations 3->3, semantic_facts 4->4 |
| hard | two distinct conflicting deal facts exist (both added as separate rows) | PASS | aw=added w=5 \| al=added l=6 |
| hard | disputed deal surfaces as pending conflict | PASS | 1 group(s) |
| hard | conflict resolved to the correct (405000) fact + loser superseded | PASS | {'winner_id': 5, 'conflict_group_id': 'cg-deal', 'superseded': [6]} |
