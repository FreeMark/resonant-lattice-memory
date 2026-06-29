"""
config_schema.py — static config field list for `hermes memory setup`.

Text/data only. Extracted verbatim from LatticeMemoryProvider.get_config_schema,
which now returns CONFIG_SCHEMA. All fields are local (no secrets/credentials).
"""

CONFIG_SCHEMA = [
    {"key": "ollama_endpoint_embed", "description": "Ollama endpoint for embeddings",
     "default": "http://localhost:11434"},
    {"key": "ollama_endpoint_reason", "description": "Ollama endpoint for the reasoning model",
     "default": "http://localhost:11434"},
    {"key": "embed_model", "description": "Embedding model name", "default": "nomic-embed-text"},
    {"key": "embed_timeout",
     "description": "HTTP timeout (seconds) for embedding calls. Default 30 (was 5) so a COLD "
                    "networked embedder — e.g. a small GPU that idle-unloaded the model — can load "
                    "on the first call instead of timing out and silently dropping facts at "
                    "consolidation time (~6s cold vs ~0.6s warm observed).",
     "default": 30.0},
    {"key": "embed_keep_alive",
     "description": "Ollama keep_alive hint sent with every embedding request so the embed model "
                    "stays resident between turns (far fewer cold loads). e.g. '10m', '1h', '-1' "
                    "to pin indefinitely; '' to disable the hint.",
     "default": "10m"},
    {"key": "reason_model",
     "description": "Reasoning model for consolidation/abstraction (async/off the hot "
                    "path — favour quality). Exact Ollama tag (namespaced). Ideal is fully "
                    "local. When local inference is limited, fast cloud models (e.g. "
                    "deepseek-v4-flash:cloud — excellent speed + quality, very low usage) "
                    "are recommended for the memory layer. Local step-up: ibm/granite4.1:30b "
                    "Q4_K_M, never below Q4.",
     "default": "ibm/granite4.1:8b"},
    {"key": "reason_timeout",
     "description": "HTTP timeout (seconds) for reason-model calls (extraction/abstraction/"
                    "distillation). Default 300 (was 180). Runs off the hot path (dream cycle), so a "
                    "flagship reasoner (e.g. nemotron-3-ultra) that thinks past 180s won't time out "
                    "the epoch and store 0 facts. A one-off timeout is non-fatal — the next cycle retries.",
     "default": 300.0},
    {"key": "hrr_dim", "description": "HRR phase-vector dimension", "default": 1024},
    {"key": "reflection_frequency", "description": "Consolidate every N turns", "default": 5},
    {"key": "dream_every_n_consolidations", "description": "Dream cycle every N consolidations",
     "default": 2},
    {"key": "short_tier_cycles", "description": "Dream-cycle dwell before short→mid promotion",
     "default": 3},
    {"key": "mid_tier_cycles", "description": "Dream-cycle dwell before mid→long promotion",
     "default": 6},
    {"key": "promotion_resonance_threshold", "description": "Min resonance to promote a tier",
     "default": 4},
    {"key": "similarity_threshold", "description": "Cosine similarity for semantic dedup (0-1)",
     "default": 0.78},
    {"key": "reinforce_on_recall", "description": "Bump resonance when a fact is recalled",
     "default": True},
    {"key": "recall_bump", "description": "Resonance added per recall (gated once/cycle/fact)",
     "default": 0.34},
    {"key": "reinforce_threshold",
     "description": "Cosine >= this folds a new fact into an existing one "
                    "(near-identity; below it, store separately)",
     "default": 0.95},
    {"key": "conflict_decay_floor",
     "description": "Resonance floor during conflict bleed (0.0 = lethal duel; "
                    ">0 = non-lethal, resolve via feedback)",
     "default": 0.0},
    {"key": "max_long_facts",
     "description": "Cap on long-tier facts (0 = unlimited). Evicts weakest when exceeded.",
     "default": 1000},
    {"key": "forget_after_dormant_cycles",
     "description": "Buried-but-pluckable forget policy: cycles a fully-faded fact (resonance 0) "
                    "stays DORMANT — kept and still pluckable by a strong contextual cue — before "
                    "it is truly deleted. >0 = demote then deep-delete (the default; 'eventually "
                    "fades, preserve the essence'); 0 = delete immediately at resonance 0 (legacy); "
                    "<0 = never delete (pure archive). Cycle-driven, no wall-clock.",
     "default": 100},
    {"key": "conflict_limbo",
     "description": "Conflict-limbo (default true): a CONTESTED fact (in an active conflict group) "
                    "is held in sustained-resonance limbo — protected from decay AND prune, and "
                    "auto-bleed is skipped — so a contested belief never fades before the USER "
                    "arbitrates it (resolve_conflict); it stays flagged on recall to nudge "
                    "arbitration. False restores the original auto-bleed-to-resolution duel.",
     "default": True},
    {"key": "surprise_decay_discount",
     "description": "Surprise/importance-weighted retention (A11, 0..1, default 0.5): a fact that "
                    "ever mattered (high max_resonance_seen — a surprising one-off that entered high "
                    "via novelty_boost, or a reinforced fact) fades SLOWER (up to this fraction less "
                    "decay once its peak reaches promotion_threshold), so a unique one-off is "
                    "retained longer before going dormant. 0 = uniform decay.",
     "default": 0.5},
    {"key": "procedural_seed",
     "description": "P3e tool-grounding seed: a list of durable procedural/guardrail fact strings "
                    "ingested at startup (category=procedural, tier=long, high resonance) so the "
                    "agent is grounded from day one — e.g. the Stripe Link CLI guardrails. Idempotent. "
                    "Phrase POSITIVELY ('always require human approval') rather than naming a "
                    "forbidden capability ('never auto_approve') — the negative form primes small "
                    "models to do it. Empty = no seed.",
     "default": []},
    {"key": "keep_superseded",
     "description": "When a conflict loser is bled to 0, retire it as tier='superseded' "
                    "history (superseded_by=winner) before pruning, instead of deleting it "
                    "(Phase 1b). Superseded rows are excluded from recall/promotion/conflict "
                    "but kept as belief-history. False restores the original lethal-delete duel.",
     "default": True},
    {"key": "max_superseded_history",
     "description": "Cap on retained superseded (conflict-history) facts; the oldest beyond "
                    "the cap are dropped each dream cycle. Only used when keep_superseded=True.",
     "default": 2000},
    {"key": "freshness_halflife_cycles",
     "description": "Phase 2: cycles for a fact's confirmation freshness to halve in the "
                    "gentle recall ranking nudge (a fresh near-match can edge out a stale "
                    "strong one). 0 disables the nudge (pure similarity ranking). Cycle-driven.",
     "default": 50},
    {"key": "surface_freshness_in_recall",
     "description": "Annotate recalled facts with '[confirmed ~N cycles ago]' so the model "
                    "holds long-unconfirmed beliefs with more doubt. Presentation only.",
     "default": True},
    {"key": "stale_decay_boost",
     "description": "Optional 'use it or lose it' extra dream-cycle decay for facts that are "
                    "BOTH weak (below promotion bar) and long-unconfirmed (0 = off). Bounded; "
                    "never drives resonance below 0.",
     "default": 0.0},
    {"key": "novelty_enabled",
     "description": "Phase 3: boost a NEW fact's starting resonance by its novelty "
                    "(1 - top similarity to existing facts) so a surprising one-shot sticks "
                    "without repetition. Deterministic and bounded.",
     "default": True},
    {"key": "novelty_boost",
     "description": "Max extra starting resonance for a fully-novel fact (scaled by novelty). "
                    "A fully-novel fact can thus passively clear the promotion threshold.",
     "default": 2.0},
    {"key": "gist_before_prune",
     "description": "Phase 4 (default OFF, heaviest/LLM-bound): before pruning, cluster "
                    "dying-but-once-important facts and summarise each cluster into one "
                    "category='gist' fact (with provenance) so meaning survives detail loss. "
                    "Short-tier noise is never gisted.",
     "default": False},
    {"key": "gist_floor",
     "description": "Resonance at/below which a fact counts as 'dying' and eligible for "
                    "gisting before prune (0.0 = only facts already at the prune threshold).",
     "default": 0.0},
    {"key": "gist_min_peak_resonance",
     "description": "How strong a fact must have been once (max_resonance_seen) to earn a "
                    "gist, unless it is already mid/long tier. Keeps trivia from being gisted.",
     "default": 4.0},
    {"key": "gist_frequency",
     "description": "Run gist-preserving consolidation every N dream cycles (LLM cost control).",
     "default": 4},
    {"key": "gist_min_cluster_size",
     "description": "Min number of related dying facts required to form a gist (>=2 means a "
                    "lone fading fact is not gisted, only a fading THEME).",
     "default": 2},
    {"key": "gist_max_clusters",
     "description": "Max gist clusters summarised per pass (bounds LLM calls).",
     "default": 3},
    {"key": "surface_conflicts",
     "description": "Phase 6: in recall, gently flag a MATURE unresolved conflict once per "
                    "cycle so the agent can disambiguate via the pending_conflicts / "
                    "resolve_conflict tool actions, instead of leaving it all to the duel.",
     "default": True},
    {"key": "conflict_surface_min_group_age_cycles",
     "description": "How many cycles a conflict must persist before the recall nudge mentions "
                    "it (let the duel run first; don't nag the instant a conflict is detected).",
     "default": 2},
    {"key": "enable_relations",
     "description": "Phase 5a (default OFF): during consolidation, extract explicit "
                    "(subject, relation, object) triples from each new fact into the "
                    "fact_relations graph (entity-grounded patterns; bound-HRR encoded "
                    "for Phase-5b relational recall). Inferences are NEVER stored here.",
     "default": False},
    {"key": "relation_min_confidence",
     "description": "Min extraction confidence (0-1) for a triple to be stored. Entity-"
                    "grounded triples score higher; ungrounded ones are penalized below "
                    "this gate, so raising it keeps only well-anchored relations.",
     "default": 0.5},
    {"key": "relation_extract_llm",
     "description": "Augment deterministic triple extraction with an LLM pass (default "
                    "OFF; only when enable_relations is on). LLM triples run through the "
                    "same grounding + confidence gate. Adds an Ollama call per new fact.",
     "default": False},
    {"key": "relation_recall_hrr_floor",
     "description": "Phase 5b: HRR partial-binding similarity at/above which the relational "
                    "tool action surfaces a triple as a FUZZY match when it isn't an exact "
                    "graph match. ~0.69=all known slots, ~0.46=2 of 3, ~0.34=1 of 2; 0.4 "
                    "keeps strong partial-structure matches and drops single-slot noise.",
     "default": 0.4},
    {"key": "max_inference_hops",
     "description": "Phase 5c: default max chain length for the `infer` action (bounded "
                    "transitive reasoning over the triple graph). Inferences are DERIVED, "
                    "labelled, confidence-decayed per hop, and NEVER stored as facts. Kept "
                    "small — both combinatorial growth and inference uncertainty rise with hops.",
     "default": 2},
    {"key": "enable_self_model",
     "description": "Phase 7 (default OFF): maintain a deliberate self-model — a separate, "
                    "curated agent_identity store the autonomous ingest path can NEVER write "
                    "(only the primary-context set_self_model action can). When on, a curated "
                    "identity block is surfaced in the system prompt and the get/set_self_model "
                    "tool actions are enabled. The positive counterpart to gate_self_writes.",
     "default": False},
    {"key": "self_model_seed",
     "description": "Optional mapping of identity key -> value used to seed the self-model on "
                    "first run (e.g. {name: ..., role: ..., relationship_with_user: ...}). "
                    "INSERT-OR-IGNORE: never clobbers values the agent has curated since. Only "
                    "applied when enable_self_model is true.",
     "default": {}},
    {"key": "enable_narrative",
     "description": "Phase 8 (default OFF): at session end, write a one-paragraph LLM gist of "
                    "the session into the durable session_summaries table (survives episode "
                    "pruning) and surface the most recent as 'recent history' in the system "
                    "prompt — cross-session continuity of what you and the user did together.",
     "default": False},
    {"key": "narrative_keep",
     "description": "Max session summaries retained (oldest pruned). Bounds the autobiographical log.",
     "default": 30},
    {"key": "narrative_surface",
     "description": "How many of the most recent session summaries to surface in the system "
                    "prompt as recent-history context (the full log is read via the narrative action).",
     "default": 3},
    {"key": "narrative_min_episodes",
     "description": "Minimum episodes in a session before it earns a narrative summary "
                    "(skip trivially short sessions).",
     "default": 2},
    {"key": "gate_self_writes",
     "description": "Block autonomous ingest of the agent's own config/infra/identity "
                    "chatter (model, context size, IP) as user facts. Conservative "
                    "phrase denylist; leave on unless you want self-referential facts.",
     "default": True},
    {"key": "agent_can_delete",
     "description": "A21 no-delete (default false): the agent INFLUENCES memory, never destroys "
                    "it. With this off, the lattice_store 'remove' action is refused for the agent "
                    "and it is steered to feedback='unhelpful' (fade to dormancy, recoverable) + "
                    "pin (protect a vital fact). Set true only for an admin/operator context that "
                    "needs audited hard deletion.",
     "default": False},
    {"key": "verify_source_quote",
     "description": "Attest each extracted fact's source_quote against the consolidation "
                    "transcript (two-channel: fuzzy prose + exact specifics). A fabricated "
                    "numeric/entity specific DROPS the fact; an un-anchored quote is flagged. "
                    "Verdict stored in semantic_facts.quote_status.",
     "default": True},
    {"key": "quote_match_threshold",
     "description": "Fuzzy prose-match ratio (0-1) for source_quote attestation, scored "
                    "over a transcript WINDOW (~the quote's length) around the best match, "
                    "so it stays meaningful on long transcripts. Higher = stricter about "
                    "the quote being lifted verbatim; a longest-contiguous coverage floor "
                    "backstops it.",
     "default": 0.82},
    {"key": "prune_keep_sessions", "description": "Episodic sessions to retain", "default": 20},
    {"key": "episode_max_rows",
     "description": "Total cap on episodic rows, oldest deleted first (0 = unlimited)",
     "default": 0},
    {"key": "decay_per_cycle",
     "description": "Resonance bled from short/mid facts each dream cycle",
     "default": 0.5},
    {"key": "initial_resonance",
     "description": "Starting resonance for new facts (>= promotion threshold "
                    "allows passive promotion; below it, recall is required)",
     "default": 4},
    {"key": "recall_floor",
     "description": "Minimum cosine similarity for prefetch/search recall (0-1)",
     "default": 0.30},
    {"key": "abstraction_frequency",
     "description": "Run the abstraction pass every N dream cycles",
     "default": 3},
    {"key": "conflict_sim_low",
     "description": "Lower bound of HRR content similarity for conflict detection (0.0-1.0)",
     "default": 0.55},
    {"key": "conflict_sim_high",
     "description": "Upper bound of HRR content similarity for conflict detection (0.0-1.0)",
     "default": 0.90},
     # === Tool & Action Memory ===
    {"key": "enable_tool_memory",
     "description": "Log tool calls + results as raw procedural episodes "
                    "and periodically distill them into reusable "
                    "'procedural' facts during the dream cycle (no "
                    "per-call resonance; failures become avoidance rules)",
     "default": True},
    {"key": "tool_distill_frequency",
     "description": "Distill procedural facts from tool episodes every N dream cycles",
     "default": 2},
    {"key": "tool_distill_min_episodes",
     "description": "Min un-distilled episodes for a tool before it is generalized",
     "default": 4},
    {"key": "tool_distill_max_tools",
     "description": "Max distinct tools to distill per dream-cycle pass",
     "default": 8},
    {"key": "tool_distill_sample_size",
     "description": "Max episodes per tool fed to the distillation LLM",
     "default": 12},
    {"key": "tool_episode_keep",
     "description": "Cap on retained raw tool episodes (bounded log)",
     "default": 500},
    {"key": "health_check_every_n_dream_cycles",
     "description": "Log a read-only memory-health snapshot every N dream cycles "
                    "(0 disables the periodic log; the memory_audit tool action is "
                    "always available on demand). Dream-cycle-driven, no timers.",
     "default": 10},
    {"key": "health_near_cap",
     "description": "Resonance threshold for the 'near saturation cap' count in the "
                    "health snapshot (the recall bump saturates at 50).",
     "default": 49.0},
    # === Encryption (E0: encrypted-at-rest; 'blind' reserved for the HE tier) ===
    {"key": "encryption_mode",
     "description": "none = plaintext DB (default, no new deps). at_rest = whole-DB "
                    "SQLCipher encryption under a passphrase-derived key (requires "
                    "sqlcipher3 + argon2-cffi, the RESONANT_LATTICE_DB_ENCRYPTED env "
                    "signal set before launch, and the passphrase via "
                    "RESONANT_LATTICE_PASSPHRASE). blind = Tier-1 homomorphic blind store: a "
                    "fact's embedding, HRR lift, and entity set are stored as CKKS/AEAD "
                    "ciphertext (semantic_he/_hrr/_entities) under a multi-keyset HE keystore, "
                    "and recall runs homomorphically (vector + HRR, no FTS); needs openfhe + "
                    "cryptography + argon2-cffi and the passphrase. The DB itself stays "
                    "plaintext-at-rest in this build (composing with at_rest is a follow-up).",
     "default": "none"},
    {"key": "encryption_keystore_path",
     "description": "Path to the keystore sidecar (salt + KDF params + key-check; NO "
                    "secrets). Empty = '<db>.keys' beside the memory DB. The master key "
                    "is re-derived from the passphrase at runtime, never stored.",
     "default": ""},
    {"key": "blind_he_keystore_path",
     "description": "Tier-1 blind only: path to the HE keystore sidecar (CKKS public/eval "
                    "blobs + AES-GCM-WRAPPED HE secrets; NO clear secrets). Empty = '<db>.he' "
                    "beside the memory DB. The HE secret is unwrapped from the passphrase-"
                    "derived master key at runtime, never stored in the clear.",
     "default": ""},
    {"key": "blind_reconcile_batch",
     "description": "Tier-1 blind only: max facts the blind reconciliation pass mirrors into the "
                    "encrypted tables per call (end of each consolidation epoch + dream cycle), so a "
                    "first-blind-enable BACKFILL of a large existing store spreads across cycles "
                    "rather than doing thousands of HE encryptions at once. Normal operation (a few "
                    "new facts per cycle) never hits the cap. 0 = unlimited.",
     "default": 200},
    # Additional runtime defaults used by provider/retriever.
    # NOTE: recall_floor / conflict_sim_low / conflict_sim_high are already defined
    # above (Hebbian block) — do NOT re-add them here or they double-list in the
    # `hermes memory setup` wizard. Only genuinely-new keys belong below.
    {"key": "recall_limit", "description": "Max facts considered per recall", "default": 300},
    {"key": "recall_relevance_margin",
     "description": "Prefetch precision gate (A6): drop recalled facts scoring more than this below the "
                    "top relevance, so only the on-topic cluster is injected into context (cleaner context, "
                    "fewer tokens). 0 = off (inject everything above recall_floor). 0.20 was chosen "
                    "empirically as the knee — it trims the near-relevant long tail (~50 distractors in a "
                    "loaded store) while dropping ZERO relevant facts; 0.15 starts dropping relevant ones. "
                    "Applies to the AUTONOMOUS prefetch only; the explicit lattice_store 'search' action "
                    "still returns the full ranked list. Ignored by the blind/HE retriever (vector-only).",
     "default": 0.20},
    # Additional abstraction/cluster keys used in provider (centralized via DEFAULTS)
    {"key": "abstraction_max_facts", "description": "Max facts scanned per abstraction pass", "default": 180},
    {"key": "abstraction_max_clusters", "description": "Max clusters summarised per abstraction pass", "default": 6},
    {"key": "abstraction_min_cluster_size", "description": "Min facts to form an abstraction cluster", "default": 3},
    {"key": "abstraction_max_cluster_size", "description": "Max facts per abstraction cluster", "default": 8},
    {"key": "cluster_hrr_similarity", "description": "HRR similarity to cluster facts together", "default": 0.68},
    {"key": "cluster_entity_overlap", "description": "Entity overlap to cluster facts together", "default": 0.55},
    {"key": "abstraction_dedup_threshold", "description": "Cosine above which a new abstraction is a duplicate", "default": 0.82},
    {"key": "importance_decay_discount",
     "description": "Importance-weighted retention (0.0-1.0, default 0.0 = off). A fact in a high-stakes "
                    "category (see importance_categories) decays this much SLOWER per cycle, so an "
                    "important-but-rarely-recalled fact (a compliance rule, a spend record) resists fading "
                    "and reaches the decay-exempt long tier where generic noise would prune. Fixes the "
                    "'importance != frequency' gap: retention no longer depends only on use/pin/novelty. "
                    "Bounded (the long-tier cap still applies), so it does not reintroduce unbounded growth. "
                    "0.6 is a good on-value; recommended ON for financial/compliance agents.",
     "default": 0.0},
    {"key": "importance_categories",
     "description": "Categories treated as high-stakes for importance_decay_discount — facts here decay "
                    "slower so they are retained even when unused. Default covers money/compliance/policy.",
     "default": ["policy", "rule", "compliance", "guardrail", "procedural", "spend", "financial", "legal"]},
    {"key": "detect_policy_conflicts",
     "description": "Also flag entity-less POLICY contradictions during conflict detection — two "
                    "policy/rule facts about the same action with OPPOSITE stance (tighten vs loosen, "
                    "e.g. 'never auto-approve' vs 'auto-approval enabled'). Content-similarity + entity "
                    "overlap can't pair these (sim ~0, no entities), so this is a conservative lexical "
                    "path (shared action stem + opposite stance, policy-like facts only). Surfaces the "
                    "contradiction for resolution; with conflict_limbo ON a false flag only protects + "
                    "nudges, never destroys. Set False to disable the heuristic.",
     "default": True},
    {"key": "quarantine_high_stakes_conflicts",
     "description": "Conflict CONTAINMENT (default OFF; recommended ON for money/compliance agents). When a "
                    "fact is in an UNRESOLVED conflict (conflict_group_id set) AND its category is "
                    "high-stakes (see importance_categories) AND it is NOT pinned, withhold it from the "
                    "autonomous recall block and surface a [WITHHELD] notice instead of ranking it. Turns "
                    "'the right value is somewhere in top-k' into 'the agent cannot silently act on a "
                    "contested high-stakes value before it is resolved'. A PINNED member is the user-declared "
                    "authority and is never withheld; non-high-stakes conflicts are untouched (still ranked + "
                    "[CONFLICT LOCK]-tagged). Recall-path only (the explicit search action is unaffected); "
                    "the facts stay in the store and are still flagged for resolve_conflict.",
     "default": False},
]

# Central source of truth for all runtime defaults.
# Other modules (provider, store, evals, tests) should import from here.
# Phase 9: to monitor for future drift when adding keys, always extend CONFIG_SCHEMA
# and ensure tests (e.g. test_provider_and_store_produce_identical_core_defaults) cover new ones.
DEFAULTS = {item["key"]: item["default"] for item in CONFIG_SCHEMA}
