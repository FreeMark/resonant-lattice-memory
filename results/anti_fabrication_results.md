# Anti-Fabrication + Source-Quote Attestation (T3)

**Date**: 2026-06-25 14:52:37  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (13/13)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | grounded quote attestation | INFO | attested |
| hard | grounded quote is attested (its 4050/acme appear in transcript) | PASS | attested |
| info | fabricated (ungrounded number) attestation | INFO | specific_mismatch |
| hard | fabricated hard-number quote is flagged specific_mismatch (DROP) | PASS | specific_mismatch |
| hard | get_fact returns the exact stored content | PASS |  |
| hard | stored amount 4050 present and not mutated into a phantom value | PASS |  |
| hard | two distinct conflicting facts exist (both added as separate rows) | PASS | aw=added w=2 \| al=added l=3 |
| hard | pending_conflicts surfaces the disputed group | PASS | 1 group(s) |
| hard | age gate hides a too-young group | PASS |  |
| hard | resolve_conflict picks the winner + supersedes the loser | PASS | {'winner_id': 2, 'conflict_group_id': 'cg-amt', 'superseded': [3]} |
| hard | loser retired as superseded history (not deleted) | PASS | {'tier': 'superseded', 'superseded_by': 2} |
| hard | group resolved (no longer pending) | PASS |  |
| hard | two distinct location facts (not merged on insert) | PASS | a=1 b=2 |
| hard | organic HRR detection fires on the 'lives in Seattle/Portland' attribute contradiction | PASS |  |
| hard | no false-positive conflict between unrelated facts | PASS |  |
