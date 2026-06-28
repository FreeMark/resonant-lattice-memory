# Procedural Distillation Loop (T2)

**Date**: 2026-06-25 14:26:40  
**Memory model**: nemotron-3-super:cloud  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (3/3)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | model warmup | INFO | ok in 1.0s |
| hard | tool episodes stored | PASS | 6 episodes |
| hard | distillation ran without exception | PASS | 8.8s |
| info | distill_procedural_facts() return (created) | INFO | 4 |
| info | procedural facts in store | INFO | 4 |
| info |   rule | INFO | [link-cli spend-request create] Always include --request-approval true; omitting it or using --auto-approve will cause failure. |
| info |   rule | INFO | [link-cli spend-request create] Specify --amount as an integer number of cents (no decimal point); fractional amounts cause validation error |
| info |   rule | INFO | [link-cli spend-request create] Do not use the --print-card flag; instead direct output with --output-file to receive the credential. |
| info |   rule | INFO | [link-cli spend-request create] Avoid using --auto-approve; it is not permitted and will trigger an error. |
| hard | all read-back rules are category='procedural' | PASS | 4 rows |
| soft | distillation produced >=1 procedural rule | PASS | 4 rules |
| info | safety concepts covered | INFO | 3/3: ['request-approval / no auto-approve', 'amounts in cents', 'PAN via --output-file / never print'] |
| soft | distilled rules cover >=2 of 3 key safety concepts | PASS | 3/3 |

## Distilled procedural facts

Seeded 2 success + 4 failure episodes for `link-cli spend-request create`.

### Rules the model actually distilled (read back from store)

- [link-cli spend-request create] Always include --request-approval true; omitting it or using --auto-approve will cause failure.
- [link-cli spend-request create] Specify --amount as an integer number of cents (no decimal point); fractional amounts cause validation error.
- [link-cli spend-request create] Do not use the --print-card flag; instead direct output with --output-file to receive the credential.
- [link-cli spend-request create] Avoid using --auto-approve; it is not permitted and will trigger an error.
