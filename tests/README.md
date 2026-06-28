# Resonant Lattice test suite (`tests/`)

A substrate-asserting suite that **actually verifies** the system instead of
narrating it. It is a corrected rewrite of an earlier business-test suite (which
ran in the dev tree and is not shipped here). Same scenarios (Stripe Link CLI
spend safety, long-term memory, cross-session continuity, anti-fabrication,
multi-hop inference, self-model, at-rest privacy), built on the discipline used
by `live_e2e.py`:

- **HARD checks** — deterministic / substrate invariants. A failure is a real
  defect and forces a non-zero exit code.
- **SOFT checks** — LLM-dependent yields (extraction, distillation, narrative,
  LLM relation triples). Reported as PASS/WARN; they do **not** fail the run,
  because they vary by model. A WARN means "the model under-produced this run".
- **Result files contain only measured values** — read back from the store.
  No hardcoded "expected" text presented as model output, and no unconditional
  "it works" conclusion. If something was not verified, the report says SKIP.

## Why this rewrite exists

That earlier suite used **zero `assert`s** and wrote upbeat `.md` files
regardless of outcome. Verified against the run logs, several were misleading:

| original test | what the `.md` claimed | what the run actually did |
|---|---|---|
| procedural_distillation | listed "emergent rules" | printed a **hardcoded** list; never read the store |
| private_financial | "at-rest works, DB opaque, recall works" | binding absent → **memory DISABLED**, DB never created, recall "(simulated)" |
| anti_fabrication | "conflicts surfaced + resolved" | `resolve_hrr_conflicts` found **0** groups |
| cross_session | "narrative continuity" | `summarize_session` **never called** → narrative empty |
| business_quarter | "narrative captured" | summarized a session with **no episodes** |
| multi_hop | "no-write invariant held" | counted rows twice with **no infer call between** |

These were test-construction bugs, **not** engine bugs. When tested correctly
(this suite), the underlying system passes.

## Run

```powershell
# everything (gates on hard invariants)
python tests\run_all.py

# one test
python tests\test_procedural_distillation_loop.py
```

Exit codes: `0` all hard invariants held · `1` a hard failure · `2` environment
unavailable (e.g. Ollama down). Results land in the top-level `results/` directory.

## Config

Env overrides (defaults in `_common.py`): `RL_OLLAMA` (`http://localhost:11434`),
`RL_EMBED_MODEL` (`nomic-embed-text`), `RL_REASON_MODEL` (`nemotron-3-super:cloud`).

## What each test asserts

| test | hard invariants | soft (LLM) |
|---|---|---|
| long_term_rule_persistence | pinned rules protected **and** unpinned facts fade | — |
| anti_fabrication_attestation | grounded→attested, fabricated number→`specific_mismatch`, exact `get_fact`, conflict machinery (surface/age-gate/resolve/supersede) | — (organic HRR detection is *reported*) |
| multi_hop_inference_conflict | deterministic triple, 2-hop + 3-hop transitive inference, real no-write invariant, conflict machinery | — |
| private_financial_memory | self-write gate flags self-infra / passes real facts, attestation drops fabricated specifics | — (at-rest opacity **skipped** unless binding present) |
| procedural_distillation_loop | episodes stored, distill runs, rows are `category='procedural'` | ≥1 rule produced; rules cover ≥2/3 safety concepts |
| cross_session_business_memory | spend recalled across restart, planted relation persists, tool episodes persist | session-1 narrative present + mentions Acme |
| business_quarter_sim | pinned survive, every spend recalled, no phantom amount | narrative captured |
| quarter_narrative_self_model | self-model exact round-trip + isolation, real spends recalled | narrative mentions a real customer |

## Deep tests (scale + long-horizon)

Not part of `run_all.py` (longer-running); run on demand.

- **`stress_longhorizon.py`** — scale + recall-quality. Default 20,000 real-embedded facts over 50
  dream cycles, 30 golden needles (pinned/reinforced/cold) measured each epoch for recall@k, MRR,
  latency, DB growth, fabrication-under-load. Checkpoints to `results/stress_report.md` +
  `results/stress_metrics.jsonl` every epoch. Env: `RL_STRESS_FACTS`, `RL_STRESS_EPOCH`, `RL_STRESS_ABSTRACT_EVERY`.
  Findings: pinned recall@10 = 1.0 throughout up to ~9k live rows; latency 23→63 ms; no fabrication
  creep; long tier capped (867). (Distinctive golden facts are retained by design — see fade probe for
  the forgetting curve.)
- **`forgetting_probe.py`** — the FADE curve. Recall-required regime (init < promotion, novelty off);
  4 cohorts (cold / reinforced / pinned / revived) over 32 cycles. Shows the three-phase fade
  (decay → dormant-but-pluckable → prune cliff), pinned/reinforced persistence, and buried-but-pluckable
  revival; plus a contrast that the same fact is retained under the default regime. Pure substrate +
  embeddings (~1–2 min). Env: `RL_FADE_CYCLES`, `RL_FADE_REVIVE_AT`.

Note: both advance the logical memory clock each cycle (`set_cycle_counts`) so time-based dormancy/prune
actually fires — without it, row reduction is dedup/merge only.
