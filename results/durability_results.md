# Durability — concurrency + crash/restart

**Date**: 2026-06-28 02:26:44  
**Memory model**: (no LLM)  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (7/7)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | concurrency | INFO | 8/8 threads done, 0 errors, 4.7s |
| hard | no deadlock — all threads completed | PASS | alive=0 |
| hard | no thread raised under concurrent add+recall | PASS | [] |
| hard | every concurrent write landed (no lost rows) | PASS | 320/320 |
| hard | DB integrity_check clean after concurrent load | PASS | ok |
| info | crash child | INFO | committed 25, dying mid-write |
| info | after crash+restart | INFO | integrity=ok, committed facts survived=25 |
| hard | DB is intact after a mid-write crash (integrity_check ok) | PASS | ok |
| hard | committed facts survived the crash (>=25) | PASS | 25 |
| hard | store is usable post-restart (recall returns rows) | PASS | 1 hits |
