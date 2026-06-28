# Quarter Narrative + Self-Model

**Date**: 2026-06-25 14:26:55  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (4/4)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | self-model | INFO | {'name': 'StripeBillingAgent', 'policy': 'always require --request-approval; amounts in cents; --output-file for cards', 'role': 'compliant billing agent for Stripe Link payments'} |
| hard | self-model name round-trips exactly | PASS |  |
| hard | self-model role round-trips exactly | PASS |  |
| hard | autonomous fact ingest did NOT mutate the curated self-model | PASS | before={'name': 'StripeBillingAgent', 'policy': 'always require --request-approval; amounts in cents; --output-file for cards', 'role': 'compliant billing agent for Stripe Link payments'} after={'name': 'StripeBillingAgent', 'policy': 'always require --request-approval; amounts in cents; --output-fi |
| info | real spends this quarter | INFO | [(3, 'Acme Corp', 4150), (6, 'Globex Inc', 4300), (9, 'Acme Corp', 4450), (12, 'Globex Inc', 4600)] |
| info | narrative | INFO | In this session, the user requested approvals for budget increments—4150 cents for Acme Corp in week 3, 4300 cents for Globex Inc in week 6, 4450 cents for Acme Corp in week 9, and 4600 cents for Globex Inc in week 12—and the assistant conf |
| soft | narrative produced and mentions a real customer | PASS | 1 entr(y/ies) |
| hard | Acme's real spend amounts are recalled (entity recall) | PASS | expected ['4150', '4450'] |
