# Resonant Lattice — Memory Evolution Roadmap

A phased plan to grow the system from an excellent *retrieval* memory into one that
has a sense of **time**, **self**, and the ability to **reason** over what it knows.
Eight phases, ordered by importance. We work them one at a time, each as its own
test-backed commit, exactly like the refactor + hardening passes.

---

> **STATUS: COMPLETE ✅ (2026-06-18).** All eight phases shipped as separate,
> test-backed, substrate-validated commits on `master`, each gated behind a
> default-OFF flag **where it changes behaviour**. Final state: 47 tests green,
> `LatticeStore` 90 / provider 36 methods. Phase → commit: P1 `07f9636`/`0d2870a`,
> P2 `d825327`, P3 `e29f875`, P4 `728a5b4`, P5 `4c46ccc`/`9ca3e18`/`d11473a`,
> P6 `c481ec6`, P7 `5d52333`, P8 `32e5b44` (+ post-validation polish `2df0707`).
> 
> **Default flags**: Core (P1–P3 + P6) are ON by default. Heavier phases (P4 gist, P5 relations, P7 self-model, P8 narrative) are OFF by default for safety/LLM cost control, but recommended ON via `recommended_config.yaml`.
> Recommended flags are wired ON in `recommended_config.yaml`; per-module detail is
> in `MODULE_MAP.md`. Validated live end-to-end on local models (e.g. gemma4, granite) and fast cloud models
> (e.g. deepseek-v4-flash:cloud). The plan below is preserved as the design record. **Next: the
> encryption north star (a separate, larger effort) — not part of this roadmap.**

---

## Cross-cutting principles (apply to every phase)

1. **Cycle-driven, never wall-clock.** `memory_cycle` (and `dream_cycle`) are the
   logical clock. They already persist in `meta` via `get_cycle_counts` /
   `set_cycle_counts`. No `time`, `datetime`, or timers leak into the model.
2. **Idempotent, self-detecting migrations.** Every schema change is a new
   `_migrate_*` in `store_schema.py` (SchemaMixin), registered in
   `_migrate_schema()`, guarded by a `PRAGMA table_info` / `sqlite_master` check.
   New columns are nullable with sane defaults; legacy DBs upgrade cleanly.
3. **Anti-fabrication is sacred.** New write paths inherit the existing discipline:
   inferences and gists are **clearly labelled** and never masquerade as verbatim
   facts; the `source_quote` / `quote_status` machinery extends to them; autonomous
   paths stay conservative (under-block over drop).
4. **Land on the mixin structure.** Each change names its target module. Store-side
   logic → the relevant `store_*` mixin; provider-side → `consolidation.py` /
   `recall.py` / `tool_handler.py` / `lifecycle.py`. Shared leaf code → `store_common.py`.
5. **Behaviour behind a flag.** Each phase ships behind a config key (default chosen
   per risk) so it can be rolled back without code changes.
6. **Substrate-validated acceptance.** Every phase defines (a) the exact `sqlite3`
   query that proves it at the row level, and (b) tests. The acceptance suite
   (`test_resonant_lattice.py` plus the end-to-end `tests/live_e2e.py`) extends per phase.

### Per-phase rhythm (the repeatable checklist)
`baseline green → migration → store method(s) → provider wiring → tool/recall
surface → config key(s) → tests + substrate query → inventory/verify_logic update
→ commit`. Stop after each phase for review before starting the next.

### Dependency graph
```
P1 Temporal ──► P2 Freshness
   │  └────────► P4 Gist-forgetting (uses tier history + supersedion)
   └───────────► P6 Conflicts-as-conversation (uses supersedion to resolve)
P3 Salience ──► (feeds P4: "was this ever important")
P5 Relational ── (independent)
P7 Self-model ── (independent)
P8 Narrative ─── (independent; pairs well with P4/P8 gisting infra)
```

---

## Phase 1 — Temporal & supersedion layer  *(highest importance)*

**Goal.** Facts know *when* they were learned, *when* last confirmed, and *what
replaced them* — turning a state store into something that can reason about change.

**Why.** A memory is the history of what you believed, not just the current row.
"You've preferred dark themes since cycle 40; before that, light" is the difference
between a database and a memory. The clock already exists (`memory_cycle`); we just
stamp it.

**Depends on.** Nothing. Enables P2, P4, P6.

**Data model** — migration `_migrate_add_temporal` (SchemaMixin):
- `semantic_facts.learned_at_cycle INTEGER` — memory_cycle at first INSERT.
- `semantic_facts.last_confirmed_cycle INTEGER` — memory_cycle at last reinforcement.
- `semantic_facts.superseded_by INTEGER` — id of the fact that replaced this one
  (nullable; `ON DELETE SET NULL`).
- `semantic_facts.superseded_at_cycle INTEGER` — when supersedion happened.
- Index on `superseded_by` (partial, `WHERE superseded_by IS NOT NULL`).

**Code touchpoints.**
- `store_facts.py`: a tiny `_current_memory_cycle()` helper reading `meta`
  (single source of truth — no caller signature changes). `add_or_reinforce_fact`
  stamps `learned_at_cycle = last_confirmed_cycle = current` on INSERT, and updates
  `last_confirmed_cycle = current` on reinforce (semantic/exact).
- `store_dream.py`: change the conflict end-game so a pruned loser is **superseded,
  not deleted** — in `prune_weak_facts` (or a new `supersede_conflict_losers` run
  before it), set `superseded_by` = the surviving group member + `superseded_at_cycle`,
  and move the row to a terminal `tier='superseded'` (excluded from recall/promotion)
  rather than `DELETE`. Keep a bounded history (config cap).
- `store.py`/`get_fact`: return the temporal fields.
- `tool_handler.py`: new read-only action `fact_history(fact_id)` that walks the
  `superseded_by` chain.

**Config.** `keep_superseded` (default True), `max_superseded_history` (cap, default 2000).

**Anti-fabrication / invariants.** Superseded rows are excluded from recall and
promotion (a new `WHERE tier != 'superseded'` clause in the recall/promote paths).
They are history, not active belief. No content is rewritten.

**Acceptance.**
- Substrate: `SELECT id, content, learned_at_cycle, last_confirmed_cycle,
  superseded_by, superseded_at_cycle FROM semantic_facts;`
- Tests: insert at cycle 0, bump cycle to 3, reinforce → `last_confirmed_cycle == 3`.
  Create a conflict, resolve a dream cycle → loser row still present with
  `superseded_by = winner_id`, `tier='superseded'`, excluded from `search`.

**Risks.** Reading `meta` per insert (cheap; one indexed lookup). Supersedion changes
the conflict end-game — gate behind `keep_superseded` and keep the bound.

**Complexity.** Medium. (1a columns+stamping is small/low-risk; 1b supersedion is the
careful part — ship 1a first, then 1b.)

---

## Phase 2 — Decouple strength from freshness

**Goal.** Recall and confidence consider *recency of confirmation*, not just resonance.

**Why.** `resonance_count` currently conflates importance and recency. A strong but
long-unconfirmed belief should be held with calibrated doubt.

**Depends on.** P1 (`last_confirmed_cycle`).

**Data model.** None (derived: `staleness_cycles = current_cycle - last_confirmed_cycle`).

**Code touchpoints.**
- `retrieval.py` (`search`): add an optional freshness term to ranking — a soft
  penalty on very stale facts so a fresh near-match can edge out a stale strong one.
  Keep it gentle (rank nudge, not a hard filter).
- `recall.py` (`_compute_prefetch`): annotate each recalled fact with a freshness /
  confidence hint, e.g. `[confirmed ~N cycles ago]`, so the model self-calibrates.
- `store_dream.py` (optional): a "use it or lose it" nudge — facts that are both
  low-resonance **and** long-unconfirmed decay slightly faster.

**Config.** `freshness_halflife_cycles` (soft confidence decay, default e.g. 50),
`surface_freshness_in_recall` (default True), `stale_decay_boost` (default 0 = off).

**Anti-fabrication / invariants.** Presentation/ranking only; never deletes or rewrites.

**Acceptance.**
- Substrate: compare `last_confirmed_cycle` to current cycle for two facts.
- Tests: a fact confirmed long ago carries a higher staleness annotation; ranking
  reflects the freshness nudge between an equally-similar fresh vs stale fact.

**Risks.** Over-penalising stale-but-correct facts — keep the nudge soft and tunable.

**Complexity.** Low–Medium.

---

## Phase 3 — Salience / novelty at ingestion

**Goal.** Novel, surprising facts enter at higher resonance so important one-shot
facts stick without needing repetition.

**Why.** "Your daughter's name is Maya" should survive on first mention; "nice weather"
should not. Surprise is exactly what biological memory privileges — and we already
compute the surprise signal.

**Depends on.** None (synergises with P4).

**Data model.** Optional `semantic_facts.max_resonance_seen REAL` (helps P4 know
"was this ever important"); migration `_migrate_add_salience`.

**Code touchpoints.**
- `store_facts.py` (`add_or_reinforce_fact`): on a fresh INSERT, reuse the
  `_find_semantic_match` top-1 similarity already computed. `novelty = 1 - top_sim`
  (no match ⇒ fully novel). `effective_initial = initial_resonance + novelty_boost * novelty`.
  Update `max_resonance_seen` on every reinforce.
- (Phase 3b, optional) the consolidation extraction prompt can emit a coarse
  `importance` hint for a small, conservative boost — but novelty is the deterministic
  default; LLM importance is opt-in to avoid inflation.

**Config.** `novelty_enabled` (default True), `novelty_boost` (max extra resonance for
a fully-novel fact, default ~2.0).

**Anti-fabrication / invariants.** Deterministic and bounded; a novel fact can now
clear `promotion_threshold` passively (the intended effect — document the interaction
with the existing `initial_resonance < promotion_threshold` warning).

**Acceptance.**
- Substrate: `SELECT content, resonance_count FROM semantic_facts ORDER BY id DESC;`
- Tests: a fact unlike anything stored gets higher initial resonance than one near an
  existing fact; `max_resonance_seen` tracks the peak.

**Risks.** Novelty boost interacting with promotion — covered by tests + the warning.

**Complexity.** Low.

---

## Phase 4 — Gist-preserving forgetting

**Goal.** Before pruning a fading fact, preserve its *meaning* (gist) so detail loss
isn't meaning loss — mirroring hippocampal→neocortical consolidation.

**Why.** Today `prune_weak_facts` deletes wholesale; meaning only survives if the fact
happened to be in an abstraction cluster. Human memory degrades gracefully.

**Depends on.** P1 (supersedion/tier history), P3 (`max_resonance_seen`). Reuses the
abstraction infrastructure.

**Data model.** `category='gist'` facts (no new table); provenance via the existing
`abstraction_sources` table (gist ← dying sources).

**Code touchpoints.**
- `store_dream.py` / `store_abstraction.py`: a new `consolidate_before_prune()` step
  run in `_run_dream_cycle` *before* `prune_weak_facts`. It selects dying facts
  (`resonance_count <= gist_floor`) that (a) are **not** already represented in an
  abstraction and (b) *earned their place once* (`tier IN ('mid','long')` or
  `max_resonance_seen >= threshold`), clusters them (reuse the HRR/entity clustering),
  LLM-summarises each cluster into one `gist` fact, records provenance, then lets the
  originals prune. Short-tier noise is **not** gisted.

**Config.** `gist_before_prune` (default False until validated), `gist_floor`,
`gist_min_peak_resonance`, frequency in dream cycles.

**Anti-fabrication / invariants.** Gist facts are clearly categorised (`gist`) and
carry provenance to their (now-pruned) sources; framed as summary, not verbatim.

**Acceptance.**
- Substrate: a long-tier fact decayed to ~0 → a `category='gist'` row exists, the
  original is gone, `abstraction_sources` links the gist to it.
- Tests: drive a fact's resonance to 0, run the pre-prune consolidation, assert the
  gist exists and the original is pruned.

**Risks.** LLM cost; gisting noise. Mitigated by conservative gating (earned-its-place
only) and frequency control. The highest-complexity phase — ship behind a default-off flag.

**Complexity.** High.

---

## Phase 5 — HRR for relational reasoning

**Goal.** Use the holographic substrate for relational/transitive recall, not just
similarity — the door to "far exceeds current systems."

**Why.** `holographic.py` already supports bind/unbind; today it's used only as a
similarity/conflict signal (~20% of its purpose). HRR exists *for* variable binding.

**Depends on.** None (but benefits from the entity graph already present).

**Data model.** `fact_relations(fact_id, subject, relation, object, confidence)` —
migration `_migrate_add_relations` + indexes on subject/object.

**Code touchpoints.**
- **5a — extraction:** during consolidation, a constrained pass extracts simple
  `(subject, relation, object)` triples from fact content (lightweight patterns first;
  optional LLM triple pass). Store in `fact_relations`. Encode triples as bound HRR
  structures alongside the existing fact vector.
- **5b — relational query:** a `relational_recall(query)` path resolving "who/where/
  what" questions via the triple graph (SQL) and HRR unbinding for fuzzy matches.
- **5c — bounded transitive inference:** chain triples (≤ `max_inference_hops`) to
  surface *inferred* candidates (user→works-at→Acme, Acme→in→Seattle ⇒ user near
  Seattle), returned as **labelled inferences, never stored as facts**.
- Surface via a new `relational` tool action and/or enriched recall.

**Config.** `enable_relations` (default False), `max_inference_hops` (default 2),
`relation_min_confidence`.

**Anti-fabrication / invariants.** This is where discipline matters most: inferences
are surfaced as inference (distinct tag, never `quote_status='attested'`), never
written back as facts. Extraction confidence gates noisy triples.

**Acceptance.**
- Substrate: `SELECT * FROM fact_relations;` and a known transitive chain query.
- Tests: triple extraction on a crafted fact; a 2-hop inference returns a labelled
  candidate and does **not** create a stored fact.

**Risks.** Extraction quality (garbage triples), inference masquerading as fact.
Staged (5a→5b→5c) so each sub-step is independently valuable and verifiable.

**Complexity.** High (most ambitious).

---

## Phase 6 — Conflicts as conversation

**Goal.** Surface unresolved conflicts for active disambiguation instead of resolving
them silently.

**Why.** The duel-to-the-death resolves internally; the most *human* move is to ask
"I've got conflicting info on where you live — which is right?"

**Depends on.** P1 (supersedion, to record the resolution).

**Data model.** None (reuses `conflict_group_id`).

**Code touchpoints.**
- `tool_handler.py`: read-only `pending_conflicts` action returning active conflict
  groups + their competing facts; and a `resolve_conflict(winner_id)` action that
  boosts the winner and supersedes the loser (P1).
- `recall.py` / system prompt: optional gentle nudge when a conflicted fact surfaces
  ("unresolved conflicting memory about X — consider confirming"), gated so the duel
  runs a little first.

**Config.** `surface_conflicts` (default True), `conflict_surface_min_group_age_cycles`
(don't nag immediately).

**Anti-fabrication / invariants.** Resolution is user/agent-driven and explicit;
nothing is silently overwritten beyond the existing duel.

**Acceptance.**
- Substrate: create a conflict → `pending_conflicts` lists it; `resolve_conflict`
  sets winner resonance up and loser `superseded_by`.
- Tests: the full create→surface→resolve loop.

**Risks.** Nagging fatigue — the age gate + a per-conflict "asked once" guard.

**Complexity.** Low–Medium.

---

## Phase 7 — Deliberate self-model

**Goal.** A curated, never-auto-ingested store of the agent's identity, capabilities,
and standing relationship with the user — the *positive* counterpart to the Phase-E
suppression gate.

**Why.** The self-write gate only *suppresses* accidental self-chatter; there's no
place for deliberate self-knowledge. A permanent agent should know itself on purpose.

**Depends on.** None.

**Data model.** A separate `agent_identity(key, value, updated_cycle)` table (kept out
of `semantic_facts` so it can never be reached by autonomous ingest), or a strictly
write-gated `category='self_model'`.

**Code touchpoints.**
- `store.py` / a small `store_identity` helper: read/write the identity store.
- `tool_handler.py`: explicit `set_self_model` / `get_self_model` actions (primary
  context only, deliberate).
- `__init__.py` `system_prompt_block`: surface a curated identity block from the store
  instead of via fuzzy recall.
- Seeding from config.

**Config.** `enable_self_model` (default False), seed values.

**Anti-fabrication / invariants.** **Hard rule:** the consolidation/ingest LLM paths
can *read* but never *write* the self-model. Writes are explicit, primary-context only
— so it can't become a backdoor for the self-chatter the gate exists to prevent.

**Acceptance.**
- Substrate: `SELECT * FROM agent_identity;` set via the action.
- Tests: an identity entry set by the action; assert a consolidation epoch never
  writes to the identity store.

**Risks.** Backdoor risk — covered by the read-only-for-autonomous-paths invariant + test.

**Complexity.** Medium.

---

## Phase 8 — Narrative / autobiographical layer

**Goal.** Cross-session continuity of "what we've been doing together" — story, not
just atomic facts.

**Why.** Episodes are L1/ephemeral (pruned by session window); semantic facts are
atomic. There's no durable thread of *what happened across sessions*.

**Depends on.** None (reuses the session-end consolidation + gisting infra from P4).

**Data model.** `session_summaries(session_id, summary, started_cycle, ended_cycle,
created_cycle)` — migration `_migrate_add_session_summaries`; or `category='narrative'`
facts that survive episode pruning.

**Code touchpoints.**
- `lifecycle.py` (`on_session_end`) / `consolidation.py`: after final consolidation,
  generate a one-paragraph session gist via the LLM and store it (durable, survives
  episode pruning).
- `recall.py` / system prompt: surface recent narrative on session start as "recent
  history" context.
- Bounded (`narrative_keep`); old summaries can themselves abstract/decay.

**Config.** `enable_narrative` (default False), `narrative_keep` (default ~30).

**Anti-fabrication / invariants.** Narrative is explicitly framed as summary/gist, not
verbatim; bounded; never asserted as exact quotes.

**Acceptance.**
- Substrate: end a session → a narrative summary row exists with cycle stamps.
- Tests: simulate session end → narrative summary present and bounded.

**Risks.** LLM cost (already at session end), summary quality — keep it short + bounded.

**Complexity.** Medium.

---

## Sequencing & milestones

- **Milestone A — "Memory has time" (P1, P2, P3).** The highest-leverage block; small
  schema, big behavioural change. After this the system tracks *when* and *how
  strongly/freshly* it believes things, and important one-shot facts stick.
- **Milestone B — "Memory forgets gracefully & talks back" (P4, P6).** Graceful
  degradation + active conflict resolution. P4 is the heaviest single phase.
- **Milestone C — "Memory reasons & knows itself" (P5, P7, P8).** The frontier:
  relational inference, a deliberate self-model, and autobiographical continuity.

Each phase ends green on the full harness (tests + inventory + verify_logic + stub
loader) and is committed separately, then we review before the next. The canonical
single-file copy is ported phase-by-phase (or at milestone boundaries) once each is
proven on the refactored tree.
