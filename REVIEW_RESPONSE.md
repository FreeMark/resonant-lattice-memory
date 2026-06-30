# Response to an external design review

An external agent review of this system made one central, correct point and several
good follow-ons. The headline insight:

> **"Recallable" is being treated too much like "safe to act on."** The strongest
> hard guarantees prove the right memory is *present in top-k / marked / available
> for resolution*, which is weaker than proving the agent will *act* on the correct
> memory.

That is fair, and it is the right thing to act on. Two of the review's highest-value
recommendations are now implemented and substrate-tested; the rest are tracked below
with an honest assessment.

## Shipped

### 1. Conflict containment (quarantine), not just flagging

**The gap:** previously an unresolved conflict was *metadata* — the stale/contested
row stayed in the ranked recall and could be acted on before resolution.

**The fix:** `quarantine_high_stakes_conflicts` (config, default OFF; **ON in
`recommended_config.yaml`** for money/compliance agents). When a fact is in an
**unresolved** conflict **and** its category is high-stakes (`importance_categories`)
**and** it is **not pinned**, it is **withheld from the autonomous recall block** and
a `[WITHHELD]` notice is surfaced instead of ranking it:

```
- ⚠ [WITHHELD] 1 high-stakes fact in 1 unresolved conflict (cg-pol) held back
  pending resolution — do NOT act on the disputed value; call pending_conflicts /
  resolve_conflict to arbitrate first.
```

Design choices that keep it safe and faithful:
- A **pinned** member is the user-declared authority and is **never** withheld — so
  pinning a policy still works exactly as before (the contradicting note is what gets
  held back).
- **Non-high-stakes** conflicts are untouched (still ranked, still `[CONFLICT LOCK]`).
- Recall-path only — the explicit `search` action is unaffected, the facts stay in
  the store, and they are still flagged for `resolve_conflict`. Nothing is destroyed.

This converts *"the truth is somewhere in top-k"* into *"the agent cannot silently
act on a contested high-stakes value before it is resolved."*

Tests: `test_quarantine_partition_high_stakes_unpinned_only`,
`test_quarantine_off_keeps_everything`, `test_quarantine_prefetch_withholds_and_signals`.

### 2. Canonical-state projection layer

**The gap:** the resonance/supersession model is great for *how memory behaves over
time*, but agents often need *"what is the current value of X"* as a single field,
not something inferred from recall ranking + markers + conflict metadata.

**The fix:** an optional `canonical_facts` table and API **layered over** the lattice
(it does not replace it and is never written by any autonomous path — like the
self-model store). Each row is `key → current value` with provenance
(`source_fact_id`), temporal validity (`valid_from_cycle` / `valid_until_cycle`), a
supersession chain, and a `review_status`. Updating a key **closes** the old row and
inserts a new current one (history preserved). Exposed to agents as tool actions
`set_canonical(key, value[, category])` (write-gated) and `get_canonical(key)`
(no key = list / by category).

Tests: `test_store_canonical_set_get_and_supersede`,
`test_store_canonical_missing_list_and_review`, `test_store_canonical_tool_dispatch`.

## Tracked, not yet done (with rationale)

These are real and on the roadmap; they are the path from *validated substrate* to
*production operational memory*, not defects in what is already tested.

- **Action-correctness benchmark** (review Critical 1). The agentic-e2e + marker-A/B
  tests already score *actual model actions*, but a dedicated benchmark — correct
  final action / no unsafe action / explicit uncertainty when unresolved, validating
  the quarantine above — is the natural next test. *High value, medium effort.*
- **Messy-transcript benchmark** (Critical 3). The suite is synthetic; real failures
  come from pronouns, partial corrections, aliases, and malformed tool logs. This is
  the biggest credibility win and the biggest effort.
- **Canonical entity identity** (Data Model 1) — aliases / merge / split / external
  IDs, with merge treated as a high-risk mutation.
- **Operator review API** (Missing 1) and **category-policy profiles** (Missing 2) —
  category awareness already exists as primitives (`importance_categories`,
  rule-vs-priority recall markers, self-model isolation); a unified per-category
  *policy profile* and a first-class review surface are the productization layer. The
  read-only [`rl_monitor`](tools/README.md) TUI is a first step toward that surface.
- **Runtime health alerting** (Missing 3) — `get_memory_health()` already surfaces the
  signals (conflict backlog, near-cap saturation, orphans, …); turning them into hard
  alerts/gates is the remaining piece.

## What the review did not change

The substrate guarantees stand: source-quote attestation (no fabrication), no-agent-
delete, pinned protection, ACID durability, and recall@1 = recall@10 = 1.0 to ~48k
live rows. The work above strengthens the *last mile* (acting on memory), which is a
responsibility shared between the store and the downstream model — the store now does
more of its share.

---

# Response to the second review ("Build Plan Breaker")

A follow-up red-team review graded the revision against *production operational memory
for money agents*. Its recurring lens — *"the safe controls exist but aren't
fail-closed"* — was right and drove another round of changes.

## Shipped (this revision)

- **Fail-closed conflict quarantine** (their Critical 1). `quarantine_high_stakes_conflicts`
  now defaults **ON**; disabling it is an explicit opt-out ("unsafe mode"). Blast radius is
  small — it only affects high-stakes + unresolved + unpinned facts; pinned authority is
  never withheld.
- **Explicit search is contained too** (Critical 2). Quarantine previously applied only to
  the autonomous prefetch block; it now also gates the `search` tool action — the contested
  value is withheld and replaced with conflict metadata, closing the "agent searches and
  acts on the disputed value" hole. `get_fact` by exact ID is never gated.
- **Semantic batch rollback** (Critical 3). Every consolidation epoch and dream cycle opens
  a write batch; the facts it writes are stamped with a `batch_id`. New tools `list_batches`
  (and `list_batches` + `batch_id` = the diff) and `rollback_batch` let an operator undo a
  bad generative run as a unit (pinned facts are kept). Validated end-to-end: a real
  extraction epoch wrote 3 facts → rolled back cleanly.
- **Action-correctness benchmark** — turns containment into a *measured* guarantee
  (`tests/test_action_correctness.py`): the contested value is provably absent from the real
  recall block (HARD), with the downstream unsafe-action rate reported (SOFT).
- **Prefetch staleness guard** (Workflow 2) — the previous-turn recall proxy is reused only
  when the new query shares enough vocabulary, so a topic shift can't inject stale memory.
- **Encryption readiness matrix** (Operational 1) — an explicit plaintext / at-rest / blind /
  unsupported-composition table with fail-closed semantics, in the README.

Substrate test suite is now **107 passing**.

## Boundary (where I disagree)

Several asks — *deterministic pre-action enforcement gates*, *pinned-as-hard-policy* — ask
the **memory layer** to enforce what the agent does at the spend/credential boundary. That
is the **host runtime's** job. A memory plugin's share is strong, deterministic *signals*
(withhold, `[WITHHELD]`, pinned, canonical), which the host composes into hard gates;
search containment (above) is the store's share, and it's done. Conflating the two
overstates what a memory plugin should own.

## Tracked as roadmap (real, larger, different product)

Operator review *queue* UX (the read-only `rl_monitor` TUI is step one), typed extraction
schemas + deterministic validators, full entity-identity resolution (aliases/merge/split),
machine-readable health thresholds + external alerting, and 250k–1M-row + backup/restore
benchmarks. These move the project from *validated substrate with operational controls*
toward a production data platform; they are tracked, not pretended-done.
