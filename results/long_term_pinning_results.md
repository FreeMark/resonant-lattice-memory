# Long-Term Rule Persistence + Pinning

**Date**: 2026-06-25 14:26:29  
**Memory model**: (no LLM - substrate only)  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (3/3)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| hard | all 6 facts persisted on insert | PASS |  |
| info | pinned final states | INFO | {4: (5.898688971996307, 'long', True), 5: (5.637227773666382, 'long', True), 6: (5.781687378883362, 'long', True)} |
| info | normal final states | INFO | {1: (0.0, 'long', False), 2: (0.0, 'long', False), 3: (0.0, 'long', False)} |
| hard | all 3 pinned policies protected (present, pinned, long, res>2) | PASS | 3/3 protected |
| hard | all 3 unpinned facts faded after 80 cycles (res<=1 or pruned) | PASS | 3/3 faded |
