# Test results

Committed evidence from real runs. Re-generate any of these by running the
corresponding test (see the repo `README.md` → "Verify it works"). Tests that
need a model use Ollama; everything is reproducible.

> **Reading all of it in one place:** [`CONSOLIDATED_RESULTS.md`](CONSOLIDATED_RESULTS.md)
> is a single-file digest of every test in `tests/` — each with a short description of
> what it exercises followed by its verbatim measured-result table. Start there if you
> want one parseable document instead of the per-test files below.

## Substrate → behaviour → scale (in `results/`)

**Trust axis** — can the agent believe a recall enough to act on it?
- `contamination_results.md` — no cross-entity contamination (right entity → right amount, under load).
- `recency_results.md` — value updates are retained + surfaced (current, not stale).
- `poisoning_results.md` — a pinned `[PRIORITY RULE]` beats a query-optimized poison; no-delete + gaming/flood resistant.
- `conflict_flagging_results.md` — value-update + entity-less policy contradictions get flagged (incl. a fresh short-tier poison).
- `precision_results.md` — top-1 relevant + precision@5 under thousands of distractors; adaptive gate.
- `anti_fabrication_results.md` — attestation drops fabricated specifics; conflict machinery surfaces disputes.
- `abstraction_fidelity_results.md` — abstraction never invents a number; gist preserves exact $/IDs.

**Retention / scale**
- `scale_ceiling_results.md` + `scale_ceiling_metrics.jsonl` — recall@1/@10 = 1.0 to ~48k live rows; sub-linear latency.
- `stress_report.md` + `stress_metrics.jsonl` — 20k facts / 50 dream cycles; bounded growth, real plateau.
- `forgetting_report.md` + `forgetting_metrics.jsonl` — the fade curve: decay → dormant-but-pluckable → prune; pinned/reinforced persist; revival works.
- `importance_decay_results.md` — high-stakes facts retained unused; generic noise still fades (selective).
- `long_term_pinning_results.md`, `business_quarter_results.md`, `cross_session_results.md`, `procedural_distillation_results.md`, `quarter_narrative_results.md`, `multi_hop_results.md`, `private_memory_results.md` — the business-robustness battery.

**Operational**
- `durability_results.md` — concurrency-safe (no lost writes, no deadlock) + crash/restart intact (ACID).

**Agent behaviour (real models in the loop)**
- `agentic_e2e_results.md` — grounded recall across a restart + rule-following + poison-resistance, scored on real model output (gemma142k + nemotron-3-super), 9/9 each.
- `marker_ab_*.md` — A/B of the recall-block authority marker: `[PRIORITY]` makes the agent obey a pinned rule 15/15 vs 8/15 with no marker, on two model families.
