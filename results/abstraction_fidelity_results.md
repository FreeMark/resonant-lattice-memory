# Abstraction / Gist Fidelity (exact $ + IDs)

**Date**: 2026-06-27 12:16:53  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (5/5)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| hard | source invoice facts stored distinctly (merge fix holds) | PASS | 5/5 |
| info | abstractions created | INFO | 1 |
| info |   abstract | INFO | Acme invoice amounts are recorded in cents and vary by service type (hosting, data egress, overage, support, onboarding). |
| hard | abstraction introduces NO fabricated/rounded number (all numbers trace to sources) | PASS | fabricated=[] |
| hard | source facts survive abstraction (exact values stay recoverable) | PASS | 5/5 sources intact |
| info | exact source amounts echoed in the abstraction | INFO | 0/10 |
| info | gists created | INFO | 1 |
| info |   gist | INFO | Acme March invoices: INV-7001 4050 cents (hosting), INV-7002 9900 cents (support), INV-7003 12500 cents (egress) |
| hard | gist introduces NO fabricated number | PASS | fabricated=[] |
| info | exact amounts preserved by the gist | INFO | 6/6 |
| hard | gist preserves the exact money amounts (>=1, ideally all) | PASS | 6/6 |
