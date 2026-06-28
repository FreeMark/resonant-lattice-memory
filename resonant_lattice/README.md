# Resonant Lattice Memory

A **neuroplastic, Hebbian long-term memory provider** for the hermes-agent framework. Instead of a flat
notes file, the agent gets a living memory that *strengthens what it uses, forgets what it doesn't,
abstracts patterns, resolves contradictions, and remembers its own history* — all driven by a **logical
cycle clock**, never wall-clock time. It runs **fully local** (SQLite + Ollama), and can optionally
encrypt itself at rest or run as a **homomorphic "blind store"** on hardware you don't fully trust.

---

## Table of contents
- [Philosophy](#philosophy)
- [How it works](#how-it-works)
- [Features](#features)
- [Encryption (two tiers)](#encryption-two-tiers)
- [Architecture](#architecture)
- [Installation](#installation)
- [The `lattice_store` tool](#the-lattice_store-tool)
- [Validation](#validation)
- [Remediation & Changes](#remediation--changes)
- [Sample configuration (all parameters)](#sample-configuration-all-parameters)

---

## Philosophy

Five principles run through every design decision:

1. **Cycles, not seconds.** Nothing is time-based. Memory advances on a *memory cycle* (consolidation)
   and a *dream cycle* (maintenance) that tick with the conversation. This is what lets small local
   models stay coherent — and what makes expensive homomorphic maintenance affordable.
2. **Local-first / sovereign.** Your memory lives in one SQLite file on your machine, embedded by a
   local model. No data leaves the box. The encryption tiers push this further: you hold the only key.
3. **Anti-fabrication is sacred.** Recalled memory is framed as *fallible candidates*, never ground
   truth. Source quotes are attested against the transcript; a fabricated specific drops the fact.
   The system would rather say "I don't have that" than invent.
4. **Neuroplastic / Hebbian.** Facts have *resonance* that grows with use (recall, reinforcement) and
   decays without it. Strong memories forget slower; novel one-shots stick; contradictions duel.
5. **Behaviour behind a flag, validated at the substrate.** Behaviour-changing features (especially the heavier LLM-bound phases) ship gated. Core features are ON by default; P4/P5/P7/P8 are OFF by default (for cost/risk control) but recommended ON via `recommended_config.yaml`. Acceptance is always a row-level DB check, never "ask the agent what it remembers."

## How it works

```
 conversation turns
        │  sync_turn()                       ┌─────────────────────────────────────────┐
        ▼                                    │  Dream Cycle (Hebbian maintenance)        │
   episodic log ──every N turns──► Consolidation ──► decay · promote · abstract · gist  │
   (raw turns)     (reasoning LLM      │     facts    conflict-duel · prune · distill    │
                    extracts facts)    │             tool-episodes → procedural facts    │
                                       ▼             └─────────────────────────────────────────┘
                              semantic facts  ◄──── recall reinforcement
                          (content + embedding +              ▲
                           HRR vector + entities)             │ prefetch()  (hybrid vector + FTS5)
                                       │                      │
                                       └──────────────► <resonant_memory> block in the system prompt
```

- **Ingestion.** Each turn is logged as an episode. Every `reflection_frequency` turns, the reasoning
  model distills **new, durable facts** from the recent transcript (with verbatim source quotes).
- **Encoding.** A fact carries a **nomic embedding** (semantic vector search), an **HRR phase vector**
  (compositional/structural similarity, conflict detection), and an **entity set** (graph links).
- **Recall.** Before a turn, a hybrid **vector + FTS5 keyword** search returns the strongest relevant
  facts as fallible candidates; recalled facts get a small Hebbian bump.
- **Dream cycle.** Periodically: resonance decays, dwell-gated tier promotion (short→mid→long),
  contradictions bleed against each other, fading-but-important clusters are summarised, similar facts
  are abstracted, raw tool calls are distilled into reusable procedures, and the stores are bounded.

## Features

**Default behaviour note**: Core capabilities (3-tier resonance, HRR, Entity graph, Temporal P1, Freshness P2, Salience P3, Conflicts-as-conversation P6, Agent control, Tool/procedural memory, Anti-fabrication) are **ON by default**.

Heavier/LLM-bound phases (Gist forgetting P4, Relational reasoning P5, Self-model P7, Narrative P8) are **OFF by default** (to control cost and risk) but are strongly recommended ON via the values in `recommended_config.yaml`.

| Area | What it does | Default |
|---|---|---|
| **3-tier resonance** | `short`/`mid`/`long` tiers; promotion needs both resonance and dwell time. Long facts are decay-exempt but capped (`max_long_facts`). | ON |
| **HRR encoding** | Holographic Reduced Representations (phase vectors) for compositional similarity, conflict detection, and relational binding. | ON |
| **Entity graph** | Entity extraction + `fact_entities` links; entity-scoped recall and overlap. | ON |
| **Temporal model (P1)** | Facts stamped with `learned_at` / `last_confirmed` cycles; conflict losers are *superseded* (kept as belief-history), not deleted. | ON |
| **Freshness (P2)** | A long-unconfirmed fact pays a gentle ranking nudge and is surfaced as `[confirmed ~N cycles ago]` so the model self-calibrates. | ON |
| **Salience (P3)** | A novel (low-similarity) fact enters at higher resonance so an important one-shot sticks without repetition. | ON |
| **Gist forgetting (P4)** | Before pruning, dying-but-once-important clusters are summarised into a `gist` so meaning survives detail loss. | OFF (recommended ON) |
| **Relational reasoning (P5)** | Extracts `(subject, relation, object)` triples; graph + HRR fuzzy relational recall; bounded transitive `infer` (never stored). | OFF (recommended ON) |
| **Conflicts as conversation (P6)** | Contradictions duel automatically, but mature unresolved ones are surfaced for explicit `resolve_conflict`. | ON |
| **Self-model (P7)** | A curated `agent_identity` store the autonomous ingest path physically cannot write — only the explicit `set_self_model` action. | OFF (recommended ON) |
| **Narrative (P8)** | A one-paragraph autobiographical session summary (survives episode pruning), surfaced as cross-session "recent history." | OFF (recommended ON) |
| **Agent control & confidence (P4)** | The agent **influences** memory, never destroys it: `pin`/`unpin` mark a fact *identity-level* — exempt from **all** forgetting (decay, prune, eviction); to retire a wrong fact it uses `unhelpful` feedback (fades to dormancy, **recoverable**), not delete. Recall surfaces the full **A22 confidence picture** — current resonance, **peak** ("ever important"), and **entry cycle**. A one-shot **re-embed migration** makes switching the embedder on an existing store turnkey. | ON |
| **Tool/procedural memory** | Tool calls are logged as episodes and distilled into reusable `procedural` facts (failures become avoidance rules). A `procedural_seed` grounds the agent from day one. | ON |
| **Anti-fabrication** | Source-quote attestation, a self-write policy gate, and fallible-candidate framing throughout. | ON |

## Encryption (two tiers)

Both tiers derive everything from a **single master passphrase** (`hermes memory setup`); the passphrase
is destroyed after setup and re-derived on demand. Selected via `encryption_mode` (default `none`).

- **Tier 0 — `at_rest`.** Whole-DB **SQLCipher** transparent encryption under a passphrase-derived
  Argon2id key. Zero functionality loss (pages decrypt only in RAM); the file is opaque without the key.
- **Tier 1 — `blind`.** A **homomorphic blind store** for running on hardware you don't fully trust: a
  fact's **embedding, HRR lift, and entity set** are stored as **CKKS/AEAD ciphertext** (`semantic_he` /
  `semantic_he_hrr` / `semantic_he_entities`), and recall runs **homomorphically** — the store computes
  cosine similarity on ciphertext it can never decrypt. Keys are held in a multi-keyset HE keystore
  (recall @ embed-dim + HRR @ 2·hrr_dim + a light maintenance context); the agent decrypts only query
  results re-encrypted to it (proxy re-encryption); the user re-derives the master for god-mode audit.
  Needs `openfhe` (Linux) + `cryptography` + `argon2-cffi`. Recall is vector/HRR-only (no FTS over
  ciphertext). See `ENCRYPTION_ROADMAP.md` for the full design + validation.

## Architecture

Flat sibling mixins, no subpackages — see **`MODULE_MAP.md`** for the full map.

- **`LatticeStore`** = composite of store mixins (`store_schema`, `store_facts`, `store_dream`,
  `store_abstraction`, `store_episodes`, `store_entities`, `store_relations`, `store_identity`,
  `store_narrative`, `store_blind`). All SQLite + numpy; no framework dependency.
- **`LatticeMemoryProvider`** = composite of provider mixins (`tool_handler`, `consolidation`,
  `recall`, `lifecycle`) + the thin `__init__.py` entry point.
- **Helper layer** (`he_crypto`, `crypto_keys`, `retrieval`, `blind_policy`, `holographic`,
  `entity_extractor`): retrieval, HRR, and the encryption engines.

## Installation

The plugin lives in your hermes plugins directory and registers itself via `register(ctx)`.

```bash
# Core (required)
pip install numpy sqlite-vec
#   sqlite-vec is the hard requirement (vector index); numpy is needed for HRR.
#   If the stdlib sqlite3 can't load extensions, also: pip install pysqlite3-binary

# Ollama models (local inference)
ollama pull nomic-embed-text          # embeddings (768-d)
ollama pull ibm/granite4.1:8b         # reasoning (consolidation/abstraction)

# Tier 0 — encrypted-at-rest (optional)
pip install argon2-cffi sqlcipher3-wheels

# Tier 1 — homomorphic blind store (optional; Linux)
pip install openfhe cryptography argon2-cffi
```

The provider declines activation (`is_available() == False`) if `sqlite-vec` is missing, so hermes falls
back cleanly. HRR degrades gracefully without numpy.

## Recommended model stack

The three model slots reward **different** strengths, so role-specialized models beat one model
everywhere (validated by benchmark — tool-grounding, abstraction-fidelity, and embedder precision/recall
on a real-fact corpus):

| Slot | Recommended | Why |
|---|---|---|
| **Primary LLM** (the agent itself) | `gemma4:12b` | Robust tool-calling under pressure; long-context; fast. Set in hermes, not here. |
| **Memory inference** (`reason_model`) | `gemma4:26b` or `qwen3.6:27b` | Reliable contextualizing **abstraction** + grounded extraction. Small models (e.g. `gemma4:12b`) can pass extraction but **fail the abstraction JSON** — don't use one here. |
| **Memory embedding** (`embed_model`) | `embeddinggemma:300m` | Best precision-per-byte (lowest poison-leak, full recall), drop-in 768-d, tiny. `qwen3-embedding:4b` for top ranking if you accept 2560-d. |

**Consolidation is off the hot path** (cycle-driven, background), so `reason_model` can point at a
**flagship cloud API endpoint** while primary inference and embeddings stay **fully local and private** —
you pay only for the occasional dream-cycle abstraction, and get flagship-quality generalization baked
into memory at a fraction of the token cost. (With the `blind` tier, that cloud call can even run over
*encrypted* memory.) The embedder choice is no longer a lock-in: changing `embed_model` triggers a
one-shot re-embed of the existing store on the next dream cycle.

## The `lattice_store` tool

The agent (and you) drive memory through one tool, `lattice_store`, with an `action`:

| Action | Purpose |
|---|---|
| `add` | Manually store a fact. |
| `search` | Hybrid recall (vector + keyword). Each hit carries the A22 confidence picture: resonance, `peak_resonance`, `learned_at_cycle`, `pinned`. |
| `get_fact` | Fetch the exact stored row by ID (authoritative; `found:false` ⇒ not stored). |
| `fact_history` | A fact's temporal/supersedion history. |
| `feedback` | `helpful`/`unhelpful` Hebbian adjustment. **There is deliberately no agent `delete`** — to retire a wrong/stale fact use `feedback: unhelpful` (it fades to dormancy, recoverable). |
| `pin` / `unpin` | Mark a fact *identity-level* / never-forget (exempt from all decay, prune, and eviction) — or release it. Protective only; never inflates resonance. |
| `request_abstraction` | Run a generalization pass **now** (cluster related long facts → contextualized abstractions), instead of waiting for the periodic dream-cycle one. |
| `force_consolidation` / `force_dream_cycle` | Run a cycle on demand. |
| `stats` / `memory_audit` | Counts and a read-only health snapshot. |
| `pending_conflicts` / `resolve_conflict` | Inspect and settle contradictions. |
| `facts_about_entity` / `entities_for_fact` / `related_entities` | Entity-graph queries. |
| `explain_abstraction` | Why an abstraction exists (its source facts). |
| `tool_history` | Recent tool episodes. |
| `relational` / `infer` | Relational recall and bounded transitive inference. |
| `get_self_model` / `set_self_model` | The curated identity store (P7). |
| `narrative` | The cross-session autobiographical log (P8). |

## Validation

Run from the repo root:

```bash
python resonant_lattice/test_resonant_lattice.py        # unit suite (97 tests) — pure SQLite/HRR substrate
python tests/live_e2e.py --model ibm/granite4.1:8b      # real store+retriever end-to-end vs Ollama
```

Acceptance is always a substrate (row-level DB / ciphertext) check. The encryption/HE tests self-skip
without `openfhe`/`cryptography`; the blind tier is node-validated separately.


---

## Remediation & Changes

This release includes a full remediation (see `REMEDIATION_PLAN.md`):

- Unified and centralized all defaults in `config_schema.DEFAULTS`.
- Cleaned legacy config key names and added deprecation warnings.
- Improved documentation of default vs recommended feature states (P4/P5/P7/P8 gated OFF by default).
- Fixed eval/test/harness drift and updated _fresh_store to use central defaults.
- Enhanced deployment hygiene (stronger excludes, .deployignore, versioned manifest, archived obsolete files).
- Centralized sys.path logic.
- Added `get_feature_status()` and `get_defaults()` for introspection.
- Added retry logic and better logging for resilience (Ollama calls, degraded/blind modes).
- Exposed more blind-tier health in `memory_audit`.
- Sanitized example data (no previous placeholder names remain).
- Added section comments for readability in long methods.
- Full harness validation passes; inventory/stub green.

See the plan for per-phase details and rationale.

---

## Sample configuration (all parameters)

All settings are **local** (no secrets/credentials live here — the encryption passphrase comes from the
`RESONANT_LATTICE_PASSPHRASE` env var or `hermes memory setup`). Put these under
`plugins.resonant_lattice` in your hermes `config.yaml`. Every key is optional; the values shown are the
**defaults**. Omit a key to take its default.

> **Phase 2 note**: All defaults are now defined in a single place (`config_schema.DEFAULTS`) so changing
> one value automatically affects provider, store, recommended config, and tests.

```yaml
plugins:
  resonant_lattice:

    # ── Models & endpoints ────────────────────────────────────────────────
    # The three slots are independent — see "Recommended model stack" below. Consolidation/
    # abstraction is off the hot path, so reason_model can point at a stronger (even cloud) endpoint
    # while primary inference + embeddings stay local. Changing embed_model on an EXISTING store is
    # turnkey: the next dream cycle re-embeds every fact to the new model (rebuilding the vector
    # index at the new dimension if needed). Force it immediately with force_dream_cycle.
    ollama_endpoint_embed: "http://localhost:11434"   # Ollama endpoint for embeddings
    ollama_endpoint_reason: "http://localhost:11434"  # Ollama endpoint for the reasoning model (may be a cloud API)
    embed_model: "nomic-embed-text"                   # embedding model (768-d; recommended: embeddinggemma:300m)
    reason_model: "deepseek-v4-flash:cloud"           # consolidation/abstraction model (fast cloud winner: excellent speed + quality; near-zero usage on low tiers)
    # Ideal is fully local. When local inference is limited, use a fast cloud model for the memory layer (off hot path).
    # Local step-up: "ibm/granite4.1:8b" or gemma4:26b / qwen3.6:27b (small models can fail the abstraction JSON)

    # ── Cycle cadence (the logical clock) ─────────────────────────────────
    reflection_frequency: 5             # consolidate (extract facts) every N turns
    dream_every_n_consolidations: 2     # run a dream cycle every N consolidations

    # ── Resonance & tiers ─────────────────────────────────────────────────
    initial_resonance: 4                # starting resonance for a new fact (>= promotion lets it promote passively)
    decay_per_cycle: 0.5                # resonance bled from short/mid facts each dream cycle (long exempt)
    promotion_resonance_threshold: 4    # min resonance to promote a tier
    short_tier_cycles: 3                # dream-cycle dwell before short→mid
    mid_tier_cycles: 6                  # dream-cycle dwell before mid→long
    max_long_facts: 1000                # cap on long-tier facts (0 = unlimited); evicts weakest when exceeded
    forget_after_dormant_cycles: 100    # P2b buried-but-pluckable: cycles a faded fact stays DORMANT (kept, pluckable by a strong cue) before deep-delete. >0 = demote-then-delete; 0 = delete at 0 (legacy); <0 = never delete
    surprise_decay_discount: 0.5        # P3/A11 (0..1): a fact that ever mattered (high peak) decays slower, so a unique one-off is retained longer. 0 = uniform decay

    # ── Recall & dedup ────────────────────────────────────────────────────
    recall_limit: 300                   # max facts considered per recall
    recall_floor: 0.30                  # min cosine similarity for a recall hit (0-1)
    similarity_threshold: 0.78          # cosine for semantic dedup (0-1)
    reinforce_threshold: 0.95           # cosine >= this folds a new fact into an existing one (near-identity)
    reinforce_on_recall: true           # bump resonance when a fact is recalled
    recall_bump: 0.34                   # resonance added per recall (gated once/cycle/fact)
    hrr_dim: 1024                       # HRR phase-vector dimension

    # ── Freshness — Phase 2 (decoupled from strength) ─────────────────────
    freshness_halflife_cycles: 50       # cycles for confirmation freshness to halve in the ranking nudge (0 = off)
    surface_freshness_in_recall: true   # annotate recall with "[confirmed ~N cycles ago]"
    stale_decay_boost: 0.0              # extra decay for weak AND long-unconfirmed facts (0 = off)

    # ── Salience — Phase 3 (novelty at ingestion) ────────────────────────
    novelty_enabled: true               # boost a novel fact's starting resonance so a one-shot sticks
    novelty_boost: 2.0                  # max extra starting resonance for a fully-novel fact

    # ── Conflicts — Phase 6 + detection ───────────────────────────────────
    conflict_limbo: true                # A9/A13: hold a CONTESTED fact in sustained-resonance limbo (no decay/prune/auto-bleed) until the USER arbitrates via resolve_conflict; false restores the auto-bleed duel
    conflict_decay_floor: 0.0           # resonance floor during the conflict duel (0 = lethal; >0 = non-lethal) — only when conflict_limbo is false
    keep_superseded: true               # retire conflict losers as tier='superseded' history instead of deleting
    max_superseded_history: 2000        # cap on retained superseded (belief-history) facts
    surface_conflicts: true             # gently flag a mature unresolved conflict in recall (once/cycle)
    conflict_surface_min_group_age_cycles: 2   # let the duel run this many cycles before nudging
    conflict_sim_low: 0.55              # lower HRR content-similarity bound for conflict detection
    conflict_sim_high: 0.90             # upper HRR content-similarity bound for conflict detection

    # ── Abstraction / generalization ──────────────────────────────────────
    abstraction_frequency: 3            # run the abstraction pass every N dream cycles
    abstraction_max_facts: 180          # max facts scanned per abstraction pass
    abstraction_max_clusters: 6         # max clusters summarised per pass
    abstraction_min_cluster_size: 3     # min facts to form an abstraction cluster
    abstraction_max_cluster_size: 8     # max facts per cluster
    cluster_hrr_similarity: 0.68        # HRR similarity to cluster facts together
    cluster_entity_overlap: 0.55        # entity overlap to cluster facts together
    abstraction_dedup_threshold: 0.82   # cosine above which a new abstraction is a duplicate

    # ── Gist-preserving forgetting — Phase 4 (default OFF; LLM-bound) ──────
    gist_before_prune: false            # summarise dying-but-once-important clusters before pruning
    gist_floor: 0.0                     # resonance at/below which a fact counts as 'dying'
    gist_min_peak_resonance: 4.0        # how strong a fact must have been once (max_resonance_seen) to earn a gist
    gist_frequency: 4                   # run gist consolidation every N dream cycles
    gist_min_cluster_size: 2            # min related dying facts to form a gist (a fading THEME, not one fact)
    gist_max_clusters: 3                # max gist clusters per pass

    # ── Relational reasoning — Phase 5 (default OFF) ──────────────────────
    enable_relations: false             # extract (subject, relation, object) triples during consolidation
    relation_min_confidence: 0.5        # min extraction confidence for a triple to be stored (0-1)
    relation_extract_llm: false         # augment deterministic triple extraction with an LLM pass (extra Ollama call)
    relation_recall_hrr_floor: 0.4      # HRR partial-binding floor for fuzzy relational matches (~0.69 all slots, ~0.34 one)
    max_inference_hops: 2               # max chain length for the `infer` action (derived, never stored)

    # ── Self-model — Phase 7 (default OFF) ────────────────────────────────
    enable_self_model: false            # maintain a curated agent_identity store (autonomous ingest can never write it)
    self_model_seed: {}                 # optional {key: value} seed applied on first run (INSERT-OR-IGNORE)

    # ── Narrative / autobiography — Phase 8 (default OFF) ─────────────────
    enable_narrative: false             # write a one-paragraph session summary at session end (survives episode pruning)
    narrative_keep: 30                  # max session summaries retained
    narrative_min_episodes: 2           # min episodes before a session earns a summary
    narrative_surface: 3                # how many recent summaries to surface as "recent history"

    # ── Episodic log ──────────────────────────────────────────────────────
    prune_keep_sessions: 20             # episodic sessions to retain
    episode_max_rows: 0                 # total cap on episodic rows, oldest first (0 = unlimited)

    # ── Tool / procedural memory ──────────────────────────────────────────
    enable_tool_memory: true            # log tool calls as episodes and distill them into 'procedural' facts
    tool_distill_frequency: 2           # distill procedural facts every N dream cycles
    tool_distill_min_episodes: 4        # min un-distilled episodes for a tool before generalizing
    tool_distill_max_tools: 8           # max distinct tools distilled per pass
    tool_distill_sample_size: 12        # max episodes per tool fed to the distillation LLM
    tool_episode_keep: 500              # cap on retained raw tool episodes
    procedural_seed: []                 # P3e: durable procedural/guardrail fact strings ingested at startup (category=procedural, tier=long) so the agent is grounded from day one. Phrase POSITIVELY ("always require human approval"), never as a forbidden capability

    # ── Anti-fabrication & agent control ──────────────────────────────────
    gate_self_writes: true              # block autonomous ingest of the agent's own config/infra/identity as user facts
    agent_can_delete: false             # A21 no-delete: the agent influences memory, never destroys it. With false, the 'remove' action is refused and steered to unhelpful feedback (+ pin). Set true only for an admin/operator context
    verify_source_quote: true           # attest each fact's source_quote vs the transcript (fabricated specific → drop fact)
    quote_match_threshold: 0.82         # fuzzy prose-match ratio for source_quote attestation (0-1; higher = stricter)

    # ── Memory-health audit ───────────────────────────────────────────────
    health_check_every_n_dream_cycles: 10   # log a read-only health snapshot every N dream cycles (0 = off)
    health_near_cap: 49.0               # resonance threshold for the 'near saturation cap' health count

    # ── Encryption (default off; see ENCRYPTION_ROADMAP.md) ───────────────
    encryption_mode: "none"             # none | at_rest (SQLCipher) | blind (homomorphic blind store)
    encryption_keystore_path: ""        # keystore sidecar (salt/KDF params/key-check; NO secrets). Empty = '<db>.keys'
    blind_he_keystore_path: ""          # blind only: HE keystore sidecar (public/eval blobs + wrapped secrets). Empty = '<db>.he'
    blind_reconcile_batch: 200          # blind only: max facts mirrored to the encrypted tables per reconcile pass (0 = unlimited)

    # ── Prompts (override the built-in defaults; the shown text IS the default) ──
    # Each is plain text; a config value overrides the corresponding DEFAULT_* in prompts.py.

    extraction_prompt: |
      Analyze this dialogue log and extract only NEW, durable facts, user preferences, or goal
      states that are EXPLICITLY supported by the log.
      GROUNDING RULES (critical — violating them corrupts memory):
      - Do NOT invent or infer specifics (names, numbers, dates, IDs, paths, settings, versions)
        that are not literally present in the log. If a detail is not in the log, leave it out.
      - For EVERY fact, include "source_quote": the shortest snippet copied VERBATIM from the log
        (character-for-character, no paraphrase) that supports the fact.
      - If the supporting turn carried a URL or identifier, also include "source_ref"; else omit it.
      - If a candidate fact has no exact supporting snippet in the log, DROP it rather than guessing.
      Output ONLY a valid JSON array of objects with keys: "content", "category", "source_quote",
      and optional "source_ref". If nothing new is learned, output an empty array [].

    consolidation_prompt: |
      You are an expert memory abstraction engine.

      Given the following group of related long-term facts, synthesize 1-2 higher-level, more
      general abstractions that capture the common theme.

      Rules:
      - Make abstractions concise but meaningful
      - Focus on the underlying principle or preference, not specific details
      - Output ONLY a valid JSON array (no extra text)
      - Each object must have keys: "content" and "category" (use "abstract")

    gist_prompt: |
      You are a memory consolidation engine for an AI agent. The facts below are FADING from memory
      (their resonance has decayed toward zero) but they mattered once. Before they are forgotten
      entirely, write ONE concise GIST that preserves their shared MEANING while letting the specific
      details go — the way human memory keeps the gist of an experience long after the details blur.

      Rules:
      - Capture the common theme/meaning, not the particulars (drop exact numbers, dates, IDs, paths)
      - Frame it as a remembered summary/generalization, NOT a verbatim fact
      - NEVER invent specifics that are not present in the facts below
      - Output ONLY a valid JSON array with a SINGLE object with keys "content" and "category"
        (use "gist"), or [] if there is no shared meaning worth keeping

    relation_prompt: |
      Extract the explicit (subject, relation, object) triples STATED in the text below. Capture only
      relationships LITERALLY present — never infer, chain, or add outside world knowledge (that is
      done elsewhere, deliberately, and never stored as fact).

      Rules:
      - relation is a short snake_case verb phrase (e.g. works_at, lives_in, prefers)
      - subject and object are the concrete entities/terms named in the text
      - Do NOT emit a triple unless both subject and object appear in the text
      - Output ONLY a valid JSON array of objects with keys "subject", "relation", "object",
        or [] if no explicit relationship is stated

    narrative_prompt: |
      Summarise the session below as ONE short paragraph of durable autobiographical memory for an AI
      agent — what the user and assistant worked on and decided together, the kind of throughline worth
      remembering at the start of the next session.

      Rules:
      - Frame it as a remembered summary, NOT a transcript or a list of turns
      - Keep the throughline; drop turn-by-turn detail and exact wording
      - NEVER invent anything not present in the log
      - One paragraph, a few sentences at most
      - Output ONLY the paragraph, with no preamble or headings

    procedural_prompt: |
      You are a procedural memory engine for an AI agent. Below are recent records of the agent calling
      one tool, each tagged SUCCESS or FAILURE with its arguments and outcome.

      Synthesize concise, REUSABLE procedural rules to help the agent use this tool better next time:
      - Argument/context patterns that tend to SUCCEED
      - Argument/context patterns that tend to FAIL (especially valuable)
      - Preconditions, gotchas, effective usage patterns

      Rules:
      - Each rule must GENERALIZE across the records, never restate one call
      - Write each rule as a standalone, actionable fact
      - If any FAILUREs occurred, include at least one failure-avoidance rule
      - Output ONLY a valid JSON array; each object has keys "content" and "category" (use "procedural")
      - Output 1-4 rules, or [] if nothing generalizable can be learned
```

> **Note on prompts:** the YAML above shows each prompt's *default* text for completeness — you only need
> to include a `*_prompt` key if you want to **override** it. The canonical defaults live in `prompts.py`.
