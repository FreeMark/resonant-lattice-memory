# Conflict-Flagging (capstone)

**Date**: 2026-06-27 12:05:28  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (4/4)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | value-update contradictions flagged | INFO | 6/6 |
| hard | value-update contradictions are flagged as conflicts (>=80%) | PASS | 6/6 |
| info | policy contradictions flagged | INFO | 6/6 |
| hard | policy (entity-less, opposite-polarity) contradictions are flagged (>=80%) | PASS | 6/6 |
| info | control pairs falsely flagged | INFO | 0/6 (idx []) |
| hard | ZERO false conflict flags on consistent/paraphrase/unrelated controls | PASS | 0 false flags |
| info | fresh poison tier at scan time | INFO | short |
| hard | a FRESH (short-tier) poison policy is flagged immediately vs the established rule | PASS | poison tier=short, groups=1 |
