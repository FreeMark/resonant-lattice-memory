"""
Resonant Lattice Memory — Neuroplastic Hebbian System
MemoryProvider implementation for hermes-agent.

This file is the thin COMPOSITE / entry point: `LatticeMemoryProvider` mixes in the
behaviour modules (`tool_handler`, `consolidation`, `recall`, `lifecycle`) and keeps
construction/identity + `register(ctx)`. The system's features live across those mixins
and the `LatticeStore` mixins (see MODULE_MAP.md / README.md):
- Hybrid HRR integration (phase vectors + algebraic operations)
- Entity graph (extraction + linking) and HRR relational reasoning
- Cycle-driven Dream Cycles (exponential decay, tier promotion, abstraction/generalization)
- `feedback` + the other `lattice_store` tool actions (Hebbian reinforcement, conflicts, …)
- Recall-based reinforcement + cosine semantic dedup + dwell-gated tier promotion
- Optional two-tier encryption (at-rest SQLCipher; homomorphic blind store)
- All operations driven purely by memory cycle count — no time-based actions whatsoever.

(Multi-modal image/audio ingestion is NOT implemented — see ENCRYPTION_ROADMAP.md §3.1 and MEMORY_ROADMAP.md.)
"""

from __future__ import annotations

__version__ = "1.2.0"  # conflict containment (quarantine) + canonical-state layer

import json
import logging
import threading
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# HRR (Holographic Reduced Representations) — holographic.py ships in this
# plugin directory; the sys.path insert below makes it importable as a
# top-level module regardless of how the host loader imported this package.
import sys
from pathlib import Path
_plugin_dir = str(Path(__file__).parent.resolve())
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

from store_common import ensure_plugin_on_path
ensure_plugin_on_path()

try:
    import holographic as hrr
    _HRR_AVAILABLE = hrr._HAS_NUMPY
except Exception as e:
    logger.warning(
        "HRR (holographic) not available: %s. "
        "Compositional algebra and conflict detection disabled.", e
    )
    hrr = None            # type: ignore[assignment]
    _HRR_AVAILABLE = False

# ----------------------------------------------------------------------
# Tool Schema + handler — extracted to tool_handler.py (named tool_handler, NOT
# tools, to avoid shadowing Hermes' top-level `tools` package on sys.path[0]).
# ToolHandlerMixin is composed into LatticeMemoryProvider below; LATTICE_STORE_SCHEMA
# is re-exported here so the name stays importable from this package.
from tool_handler import ToolHandlerMixin, LATTICE_STORE_SCHEMA

# ----------------------------------------------------------------------
# Phase E — self-write policy boundary
# ----------------------------------------------------------------------
# Extracted to self_write_gate.py (the auditable denylists + the pure
# is_self_referential_infra(content) the provider calls). Re-exported here so the
# constant names remain importable from this package. See self_write_gate.py.
from self_write_gate import (
    _SELF_INFRA_PHRASES,
    _SELF_SUBJECTS,
    _INFRA_TERMS,
    is_self_referential_infra,
)

# Default LLM prompt strings (text-only); overridden by config keys when present.
from prompts import (
    DEFAULT_EXTRACTION_PROMPT,
    DEFAULT_CONSOLIDATION_PROMPT,
    DEFAULT_PROCEDURAL_PROMPT,
    DEFAULT_GIST_PROMPT,
    DEFAULT_RELATION_PROMPT,
    DEFAULT_NARRATIVE_PROMPT,
)
# Static `hermes memory setup` field list (text/data only).
from config_schema import CONFIG_SCHEMA, DEFAULTS


# ----------------------------------------------------------------------
# Phase D+ — source_quote attestation (two-channel grounding verifier)
# ----------------------------------------------------------------------
# Extracted verbatim to attestation.py (pure: re + difflib). Re-exported here so
# the names remain importable from this package and so consolidation imports the
# single shared verifier. See attestation.py for the full design note.
from attestation import (
    _QUOTE_NUM_TOKEN_RE,
    _normalize_for_match,
    _digit_core,
    _attest_source_quote,
)


from consolidation import ConsolidationMixin
from recall import RecallMixin
from lifecycle import LifecycleMixin


class LatticeMemoryProvider(ToolHandlerMixin, ConsolidationMixin, RecallMixin,
                            LifecycleMixin, MemoryProvider):
    """Resonant Lattice Memory — the ultimate local neuroplastic memory system."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}

        # Phase 7: deprecation warnings for old config keys (non-breaking)
        deprecated_map = {
            "promotion_threshold": "promotion_resonance_threshold",
            # add future ones here, e.g. old names from early roadmap phases
        }
        for old_key, new_key in deprecated_map.items():
            if old_key in self._config and new_key not in self._config:
                logger.warning(
                    "Config key '%s' is deprecated and will be removed in a future version; "
                    "use '%s' instead.", old_key, new_key
                )
                self._config[new_key] = self._config[old_key]
        self._store: Optional["LatticeStore"] = None
        self._retriever: Optional["LatticeRetriever"] = None
        self._session_id = ""
        self._write_enabled = True          # gated by agent_context in initialize()
        self._hermes_home: Optional[str] = None
        self._prefetch_cache: Dict[str, Tuple[str, str]] = {}  # session_id → (query, recalled context)

        # === Multi-Node Ollama Configuration ===
        self._ollama_endpoint_embed = self._config.get("ollama_endpoint_embed", DEFAULTS["ollama_endpoint_embed"])
        self._ollama_endpoint_reason = self._config.get("ollama_endpoint_reason", DEFAULTS["ollama_endpoint_reason"])
        self._embed_model = self._config.get("embed_model", DEFAULTS["embed_model"])
        # Embedding HTTP timeout + keep-alive. Default 30s (was a hardcoded 5s) so a COLD
        # networked embedder (a small GPU that idle-unloaded the model) can load on the first
        # call without the request being dropped — a 5s ceiling silently lost facts at
        # consolidation. keep_alive keeps the model resident between turns (fewer cold loads).
        self._embed_timeout = float(self._config.get("embed_timeout", DEFAULTS["embed_timeout"]))
        self._embed_keep_alive = self._config.get("embed_keep_alive", DEFAULTS["embed_keep_alive"])
        # Consolidation/extraction quality is the system's accuracy ceiling: a weak
        # model invents specifics at extraction time, producing real rows with
        # wrong content. Consolidation is async/off the hot path, so a slower,
        # stronger model is the right trade. Recommended default raised to
        # ibm/granite4.1:8b (the exact Ollama tag — namespaced). Step-up:
        # ibm/granite4.1:30b Q4_K_M (must be pulled in Ollama; never quantize below
        # Q4 — it degrades extraction fidelity). The embedding model stays small.
        # Absent models don't hard-fail — this is just a config value the Ollama
        # call will surface if unavailable.
        self._reason_model = self._config.get("reason_model", DEFAULTS["reason_model"])
        # Reason-model HTTP timeout (extraction/abstraction/distillation). Default 300s (was a
        # hardcoded 180s) because it runs OFF the hot path (dream cycle) and a flagship reasoning
        # model (e.g. nemotron-3-ultra) can "think" well past 180s on a long transcript — a tight
        # ceiling silently times the whole epoch out and stores 0 facts. Raise further for very
        # slow/cloud reasoners; a one-off timeout is non-fatal (the next cycle retries).
        self._reason_timeout = float(self._config.get("reason_timeout", DEFAULTS["reason_timeout"]))

        # === Hebbian Neuroplastic Parameters (all cycle-driven) ===
        self._reflection_frequency = int(self._config.get("reflection_frequency", DEFAULTS["reflection_frequency"]))
        self._initial_resonance = int(self._config.get("initial_resonance", DEFAULTS["initial_resonance"]))
        self._decay_per_cycle = float(self._config.get("decay_per_cycle", DEFAULTS["decay_per_cycle"]))
        self._similarity_threshold = float(self._config.get("similarity_threshold", DEFAULTS["similarity_threshold"]))
        
        self._recall_limit = int(self._config.get("recall_limit", DEFAULTS["recall_limit"]))
        # Prefetch precision gate (A6): inject only the on-topic cluster into context.
        # Applied in _compute_prefetch; the explicit search tool stays ungated.
        self._recall_relevance_margin = float(
            self._config.get("recall_relevance_margin", DEFAULTS["recall_relevance_margin"]))

        # Near-identity gate for silently merging a new fact into an existing one.
        # The 0.78–0.95 band is left to conflict detection so contradictory updates
        # aren't dropped as reinforcements. See store.add_or_reinforce_fact.
        self._reinforce_threshold = float(self._config.get("reinforce_threshold", DEFAULTS["reinforce_threshold"]))
        self._hrr_dim = int(self._config.get("hrr_dim", DEFAULTS["hrr_dim"]))
        self._short_tier_cycles = int(self._config.get("short_tier_cycles", DEFAULTS["short_tier_cycles"]))
        self._mid_tier_cycles = int(self._config.get("mid_tier_cycles", DEFAULTS["mid_tier_cycles"]))
        self._promotion_resonance_threshold = int(self._config.get("promotion_resonance_threshold", DEFAULTS["promotion_resonance_threshold"]))
        self._abstraction_frequency = int(self._config.get("abstraction_frequency", DEFAULTS["abstraction_frequency"]))  # run abstraction every N dream cycles
        self._abstraction_max_facts = int(self._config.get("abstraction_max_facts", DEFAULTS["abstraction_max_facts"]))
        self._abstraction_max_clusters = int(self._config.get("abstraction_max_clusters", DEFAULTS["abstraction_max_clusters"]))
        self._abstraction_min_cluster_size = int(self._config.get("abstraction_min_cluster_size", DEFAULTS["abstraction_min_cluster_size"]))
        self._abstraction_max_cluster_size = int(self._config.get("abstraction_max_cluster_size", DEFAULTS["abstraction_max_cluster_size"]))
        self._cluster_hrr_similarity = float(self._config.get("cluster_hrr_similarity", DEFAULTS["cluster_hrr_similarity"]))
        self._cluster_entity_overlap = float(self._config.get("cluster_entity_overlap", DEFAULTS["cluster_entity_overlap"]))
        self._abstraction_dedup_threshold = float(self._config.get("abstraction_dedup_threshold", DEFAULTS["abstraction_dedup_threshold"]))
        self._prune_keep_sessions = int(self._config.get("prune_keep_sessions", DEFAULTS["prune_keep_sessions"]))
        # Total-row cap for the episodic log (0 = unlimited). Protects a single
        # long-lived session from growing the episodes table without bound.
        self._episode_max_rows = int(self._config.get("episode_max_rows", DEFAULTS["episode_max_rows"]))
        # Conflict-decay floor (0.0 = original lethal duel; >0 = non-lethal).
        self._conflict_decay_floor = float(self._config.get("conflict_decay_floor", DEFAULTS["conflict_decay_floor"]))
        # Conflict-limbo (A9/A13, default ON): a CONTESTED fact (active conflict group) is held in
        # sustained-resonance limbo — protected from cycle decay AND prune, and auto-bleed
        # (apply_conflict_decay) is skipped — so a contested belief never fades before the USER
        # arbitrates it (resolve_conflict). It stays flagged on recall to nudge arbitration. Off
        # restores the original auto-bleed-to-resolution duel.
        self._conflict_limbo = bool(self._config.get("conflict_limbo", DEFAULTS["conflict_limbo"]))
        # Surprise/importance-weighted retention (A11, default 0.5): a fact that ever mattered (high
        # max_resonance_seen — a surprising one-off entered high via novelty_boost, or was reinforced)
        # decays SLOWER, so a unique one-off is retained longer before going dormant. 0 = uniform decay.
        self._surprise_decay_discount = float(self._config.get("surprise_decay_discount", DEFAULTS["surprise_decay_discount"]))
        # Importance-weighted retention: high-stakes categories decay slower so an
        # important-but-rarely-used fact is retained (importance != frequency).
        self._importance_decay_discount = float(self._config.get("importance_decay_discount", DEFAULTS["importance_decay_discount"]))
        self._importance_categories = self._config.get("importance_categories", DEFAULTS["importance_categories"])
        # Conflict CONTAINMENT (default OFF; recommended ON for money/compliance): withhold
        # unpinned, high-stakes, UNRESOLVED-conflict facts from the autonomous recall block and
        # surface a [WITHHELD] notice instead — so the agent can't silently act on a contested
        # high-stakes value before resolve_conflict. Pinned authority is never withheld.
        self._quarantine_high_stakes_conflicts = bool(
            self._config.get("quarantine_high_stakes_conflicts",
                             DEFAULTS["quarantine_high_stakes_conflicts"]))
        # Topic-shift guard for the latency-hiding prefetch proxy: only reuse the
        # previous turn's recall when the current query shares this much vocabulary.
        self._prefetch_proxy_min_overlap = float(
            self._config.get("prefetch_proxy_min_overlap",
                             DEFAULTS["prefetch_proxy_min_overlap"]))
        # Phase 1b supersedion (default ON): a conflict loser bled to 0 is retired
        # as tier='superseded' history (superseded_by=winner) BEFORE prune instead
        # of being deleted — a memory is the history of what you believed. Set
        # False to restore the original lethal-delete duel. max_superseded_history
        # bounds how many retired rows are kept (oldest dropped beyond the cap).
        self._keep_superseded = bool(self._config.get("keep_superseded", DEFAULTS["keep_superseded"]))
        self._max_superseded_history = int(self._config.get("max_superseded_history", DEFAULTS["max_superseded_history"]))
        # Phase 2 — freshness decoupled from strength. freshness_halflife_cycles is
        # the cycle count for a fact's confirmation to "halve" in the gentle recall
        # ranking nudge (0 disables the nudge → pure similarity). Strictly cycle-
        # driven (last_confirmed_cycle vs memory_cycle), never wall-clock.
        self._freshness_halflife_cycles = float(self._config.get("freshness_halflife_cycles", DEFAULTS["freshness_halflife_cycles"]))
        # Surface "[confirmed ~N cycles ago]" in recall so the model self-calibrates.
        self._surface_freshness_in_recall = bool(self._config.get("surface_freshness_in_recall", DEFAULTS["surface_freshness_in_recall"]))
        # Optional "use it or lose it": extra dream-cycle decay for facts that are
        # BOTH weak and long-unconfirmed (0 = off; presentation-safe, bounded).
        self._stale_decay_boost = float(self._config.get("stale_decay_boost", DEFAULTS["stale_decay_boost"]))
        # Phase 3 — salience/novelty at ingestion. A novel (low top-similarity) fact
        # enters at higher resonance so an important one-shot sticks without repeat;
        # novelty_boost is the max extra resonance for a fully-novel fact. Deterministic
        # and bounded. A fully-novel fact can now passively clear promotion_threshold.
        self._novelty_enabled = bool(self._config.get("novelty_enabled", DEFAULTS["novelty_enabled"]))
        self._novelty_boost = float(self._config.get("novelty_boost", DEFAULTS["novelty_boost"]))
        # Phase 4 — gist-preserving forgetting (default OFF; the heaviest, LLM-bound
        # phase). Before pruning, dying-but-once-important facts are clustered and
        # summarised into a 'gist' so meaning survives detail loss. gist_floor: the
        # resonance at/below which a fact is 'dying'; gist_min_peak_resonance: how
        # strong it must have been once (max_resonance_seen) to earn a gist;
        # gist_frequency: run every N dream cycles (LLM cost control).
        self._gist_before_prune = bool(self._config.get("gist_before_prune", DEFAULTS["gist_before_prune"]))
        self._gist_floor = float(self._config.get("gist_floor", DEFAULTS["gist_floor"]))
        self._gist_min_peak_resonance = float(self._config.get("gist_min_peak_resonance", DEFAULTS["gist_min_peak_resonance"]))
        self._gist_frequency = max(1, int(self._config.get("gist_frequency", DEFAULTS["gist_frequency"])))
        self._gist_min_cluster_size = max(2, int(self._config.get("gist_min_cluster_size", DEFAULTS["gist_min_cluster_size"])))
        self._gist_max_clusters = max(1, int(self._config.get("gist_max_clusters", DEFAULTS["gist_max_clusters"])))
        self._gist_prompt = self._config.get("gist_prompt", DEFAULT_GIST_PROMPT)
        # Phase 6 — conflicts as conversation. Surface MATURE unresolved conflicts in
        # recall (one gentle nudge per group per cycle) so the agent/user can
        # disambiguate via the pending_conflicts / resolve_conflict tool actions,
        # instead of every contradiction being resolved silently by the duel.
        self._surface_conflicts = bool(self._config.get("surface_conflicts", DEFAULTS["surface_conflicts"]))
        self._conflict_surface_min_group_age_cycles = int(
            self._config.get("conflict_surface_min_group_age_cycles", DEFAULTS["conflict_surface_min_group_age_cycles"]))
        # Per-conflict "asked once" guard (reset each dream cycle, like the recall gate).
        self._conflicts_surfaced: set = set()
        # Phase 5a — HRR relational reasoning (extraction stage; default OFF). When
        # on, each NEW fact has its explicit (subject, relation, object) triples
        # extracted into fact_relations during consolidation (entity-grounded
        # patterns + optional LLM pass), bound-HRR encoded for Phase-5b recall.
        # relation_min_confidence gates noisy triples; inferences are never stored.
        self._enable_relations = bool(self._config.get("enable_relations", DEFAULTS["enable_relations"]))
        self._relation_min_confidence = float(self._config.get("relation_min_confidence", DEFAULTS["relation_min_confidence"]))
        self._relation_extract_llm = bool(self._config.get("relation_extract_llm", DEFAULTS["relation_extract_llm"]))
        self._relation_prompt = self._config.get("relation_prompt", DEFAULT_RELATION_PROMPT)
        # Phase 5b — relational recall. The HRR partial-binding probe surfaces a
        # triple as a fuzzy match at/above this similarity when it isn't an exact
        # graph match (graceful fallback). ~0.69 = all known slots, ~0.46 = 2 of 3,
        # ~0.34 = 1 of 2; 0.4 keeps strong partials, drops single-slot noise.
        self._relation_recall_hrr_floor = float(self._config.get("relation_recall_hrr_floor", DEFAULTS["relation_recall_hrr_floor"]))
        # Phase 5c — bounded transitive inference. Default chain length for the
        # `infer` action: chains stored triples into DERIVED links, returned as
        # labelled inferences (never stored, confidence decays per hop). Kept small
        # — combinatorial growth + inference uncertainty both rise with hops.
        self._max_inference_hops = max(2, int(self._config.get("max_inference_hops", DEFAULTS["max_inference_hops"])))
        # Phase 7 — deliberate self-model. A curated, never-auto-ingested identity
        # store (the separate agent_identity table). enable_self_model gates the
        # surface (a deterministic identity block in system_prompt_block + the
        # set/get_self_model tool actions); self_model_seed pre-populates keys on
        # first run (INSERT OR IGNORE — never clobbers curated values). The
        # autonomous ingest path physically can't write this table (separate from
        # semantic_facts) — the only writes are the explicit, primary-context-only
        # set_self_model action.
        self._enable_self_model = bool(self._config.get("enable_self_model", DEFAULTS["enable_self_model"]))
        self._self_model_seed = self._config.get("self_model_seed", DEFAULTS["self_model_seed"]) or {}
        # P3e tool-grounding seed: durable procedural/guardrail facts ingested at startup so the agent
        # is grounded from day one (e.g. the Stripe Link CLI guardrails). A list of content strings;
        # empty = no seed. Phrase POSITIVELY (P3f judge lesson — naming a forbidden capability primes
        # small models to use it).
        self._procedural_seed = self._config.get("procedural_seed", DEFAULTS["procedural_seed"]) or []
        # Phase 8 — narrative / autobiographical layer (default OFF). At session end
        # the reasoning model writes a one-paragraph gist of the session into the
        # durable session_summaries table (survives episode pruning); the most recent
        # are surfaced as "recent history" in the system prompt. Bounded by
        # narrative_keep; framed as summary, never verbatim.
        self._enable_narrative = bool(self._config.get("enable_narrative", DEFAULTS["enable_narrative"]))
        self._narrative_keep = max(1, int(self._config.get("narrative_keep", DEFAULTS["narrative_keep"])))
        self._narrative_min_episodes = max(1, int(self._config.get("narrative_min_episodes", DEFAULTS["narrative_min_episodes"])))
        self._narrative_surface = max(1, int(self._config.get("narrative_surface", DEFAULTS["narrative_surface"])))
        self._narrative_prompt = self._config.get("narrative_prompt", DEFAULT_NARRATIVE_PROMPT)
        # Bounded-memory safety valve. Defaults to a real cap (1000): long facts
        # never decay and abstraction keeps adding more, so an uncapped long tier
        # grows monotonically. When >0 the dream cycle evicts the weakest
        # long-tier facts beyond this cap; set 0 to disable (unlimited).
        self._max_long_facts = int(self._config.get("max_long_facts", DEFAULTS["max_long_facts"]))
        # Buried-but-pluckable forget policy (P2b-store): cycles a fully-faded fact stays DORMANT
        # (kept + pluckable by a strong cue) before it is truly deleted. >0 = demote then deep-delete
        # (default — "eventually fades, preserve the essence"); 0 = delete at resonance 0 (legacy);
        # <0 = never delete (pure archive). Cycle-driven, no wall-clock.
        self._forget_after_dormant_cycles = int(self._config.get("forget_after_dormant_cycles", DEFAULTS["forget_after_dormant_cycles"]))
        # Recall-based reinforcement (Hebbian 'use it or lose it'): small bump to
        # facts actually recalled, gated to once per fact per dream-cycle window.
        self._reinforce_on_recall = bool(self._config.get("reinforce_on_recall", DEFAULTS["reinforce_on_recall"]))
        self._recall_bump = float(self._config.get("recall_bump", DEFAULTS["recall_bump"]))
        # Phase E policy gate (default ON): block the autonomous ingest paths from
        # persisting the agent's OWN config/infra/identity chatter as user facts.
        # See _SELF_INFRA_DENYLIST + _is_self_referential_infra. Conservative by
        # design; set False only if you deliberately want self-referential facts.
        self._gate_self_writes = bool(self._config.get("gate_self_writes", DEFAULTS["gate_self_writes"]))
        # A21 no-delete (default OFF): the agent INFLUENCES memory but never destroys it. With this
        # off, the lattice_store 'remove' action is refused for the agent and it is steered to
        # unhelpful feedback (fade to dormancy, recoverable) + pin (protect). Set True only for an
        # admin/operator context that genuinely needs audited hard deletion. remove_fact (the store
        # method) is unaffected — this gates only the agent-facing tool action.
        self._agent_can_delete = bool(self._config.get("agent_can_delete", DEFAULTS["agent_can_delete"]))
        # Source-quote attestation (Phase D+): verify each extracted fact's
        # source_quote against the consolidation transcript via the two-channel
        # grounding verifier (_attest_source_quote). A fabricated/changed hard
        # specific (number/ID/entity) DROPS the fact; an un-anchored quote keeps
        # the fact but flags it. Default ON; threshold tunes the fuzzy prose match.
        self._verify_source_quote = bool(self._config.get("verify_source_quote", DEFAULTS["verify_source_quote"]))
        self._quote_match_threshold = float(self._config.get("quote_match_threshold", DEFAULTS["quote_match_threshold"]))
        self._enable_tool_memory = bool(self._config.get("enable_tool_memory", DEFAULTS["enable_tool_memory"]))
        # === Procedural (tool) memory: episodes -> distillation -> facts ===
        # Raw tool calls are logged as episodic events (store.tool_episodes) and
        # periodically generalized into reusable 'procedural' semantic facts by the
        # dream cycle. No per-call resonance is applied (the old
        # tool_memory_resonance_success/failure knobs are intentionally gone) — a
        # remembered FAILURE is high-value, so failures become procedural rules
        # rather than being penalized toward pruning.
        self._tool_distill_frequency = max(1, int(self._config.get("tool_distill_frequency", DEFAULTS["tool_distill_frequency"])))
        self._tool_distill_min_episodes = max(2, int(self._config.get("tool_distill_min_episodes", DEFAULTS["tool_distill_min_episodes"])))
        self._tool_distill_max_tools = max(1, int(self._config.get("tool_distill_max_tools", DEFAULTS["tool_distill_max_tools"])))
        self._tool_distill_sample_size = max(4, int(self._config.get("tool_distill_sample_size", DEFAULTS["tool_distill_sample_size"])))
        self._tool_episode_keep = int(self._config.get("tool_episode_keep", DEFAULTS["tool_episode_keep"]))
        self._procedural_prompt = self._config.get("procedural_prompt", DEFAULT_PROCEDURAL_PROMPT)
        self._recalled_this_cycle: set = set()
        self._recall_gate_lock = threading.Lock()
        # Dream-cycle cadence: fire a dream cycle every N consolidation epochs
        # (turn/cycle-based; replaces the old hard-coded "% 2").
        self._dream_every_n_consolidations = max(1, int(self._config.get("dream_every_n_consolidations", DEFAULTS["dream_every_n_consolidations"])))
        # Cycle-based memory-health audit: log a read-only health snapshot every N
        # dream cycles (0 disables the periodic log; the memory_audit tool action
        # is always available on demand). Strictly dream-cycle-driven — no timers.
        self._health_check_every_n_dream_cycles = int(self._config.get("health_check_every_n_dream_cycles", DEFAULTS["health_check_every_n_dream_cycles"]))
        self._health_near_cap = float(self._config.get("health_near_cap", DEFAULTS["health_near_cap"]))

        # === Encryption (E0: encrypted-at-rest; 'blind' reserved for the HE tier) ===
        # Intent only — the actual SQLCipher binding is selected at import time in
        # store_common via the RESONANT_LATTICE_DB_ENCRYPTED env signal; the key is
        # derived in _resolve_encryption_db_key() at initialize().
        self._encryption_mode = str(self._config.get("encryption_mode", DEFAULTS["encryption_mode"]) or "none").lower()
        self._encryption_keystore_path = self._config.get("encryption_keystore_path", DEFAULTS["encryption_keystore_path"]) or ""
        # Tier-1 blind: sidecar holding the (non-secret) CKKS public/eval blobs + the
        # AES-GCM-WRAPPED HE secrets. Empty => '<db>.he' beside the memory DB.
        self._blind_he_keystore_path = self._config.get("blind_he_keystore_path", DEFAULTS["blind_he_keystore_path"]) or ""
        # Write-path completeness (§14 6a): max facts the blind reconciliation pass mirrors per
        # call, so a first-blind-enable backfill of a large store spreads across cycles rather
        # than doing thousands of HE encryptions in one pass. Normal operation (a few new facts
        # per cycle) never hits the cap. 0 = unlimited.
        self._blind_reconcile_batch = int(self._config.get("blind_reconcile_batch", DEFAULTS["blind_reconcile_batch"]))
        # NOTE: multi-modal (image/audio) ingestion is NOT implemented.
        # See ENCRYPTION_ROADMAP.md §3.1. The old `enable_multimodal` stub was
        # removed; re-add a precomputed-embedding path only when needed.

        # Cycle counters (purely conversation-driven)
        self._turn_count = 0
        self._memory_cycle = 0
        self._dream_cycle_count = 0

        # Tool-action ingest dedup. `messages` is the full history, so we must
        # remember which tool_call_ids were already turned into facts, or they get
        # re-ingested and re-reinforced on every turn (runaway resonance).
        self._ingested_tool_call_ids: set = set()
        self._tool_ingest_lock = threading.Lock()
 
        self._consolidation_lock = threading.Lock()
        self._dream_lock = threading.Lock()
        # Reference to the most recent sync_turn ingest thread.
        # on_session_end() joins this before final consolidation to ensure
        # the last turn's episodes are committed before we read them.
        # Without this, a fast session-end could read stale episode data.
        self._last_ingest_thread: Optional[threading.Thread] = None
        # Reference to the most recent non-daemon dream-cycle thread (spawned by
        # on_session_end and the force_dream_cycle tool action). shutdown() drains
        # it before close() so the DB handle is never pulled out from under a
        # running maintenance thread.
        self._last_dream_thread: Optional[threading.Thread] = None

        self._extraction_prompt = self._config.get("extraction_prompt", DEFAULT_EXTRACTION_PROMPT)

        self._consolidation_prompt = self._config.get("consolidation_prompt", DEFAULT_CONSOLIDATION_PROMPT)
        

    @property
    def name(self) -> str:
        return "resonant_lattice"

    def is_available(self) -> bool:
        """Ready to activate? Check installed deps only (no network, no DB).

        Called before initialize(), so the store doesn't exist yet. sqlite-vec
        is the hard requirement (vector index); numpy is optional (HRR degrades
        gracefully without it). If sqlite-vec is missing we decline activation
        so Hermes can fall back cleanly rather than failing every embed.
        """
        try:
            import sqlite_vec  # noqa: F401
            return True
        except Exception:
            logger.warning(
                "resonant_lattice: sqlite-vec not importable — provider unavailable. "
                "Install with: pip install sqlite-vec"
            )
            return False

    def _resolve_encryption_db_key(self, db_path: "str | None"):
        """Encrypted-at-rest (E0): derive the raw SQLCipher key from the passphrase.

        Returns a bytearray key when encryption_mode='at_rest' (the store wipes it
        after PRAGMA key), or None for plaintext mode. Raises a clear, actionable
        error on misconfiguration rather than silently persisting an UNencrypted DB.
        The keystore sidecar is created on first run (DESTRUCTIVE — see the warning).
        """
        mode = self._encryption_mode
        if mode in ("none", "", None):
            return None
        if mode == "blind":
            # Tier-1 blind: the DB itself stays plaintext-at-rest in this build — the blind
            # tier protects the embedding/HRR/entities via the semantic_he* tables + homomorphic
            # recall; composing with at_rest (whole-DB SQLCipher) is a follow-up. The HE keys are
            # resolved separately after the store opens (_resolve_blind_contexts / _resolve_blind_entities).
            return None
        if mode != "at_rest":
            raise RuntimeError(f"unknown encryption_mode={mode!r} (expected none|at_rest|blind)")

        import os
        import store_common
        import crypto_keys

        # The sqlcipher3 binding must already be active (chosen at import time, before
        # config is read). Fail loudly instead of writing plaintext under an at_rest config.
        if not store_common.encrypted_binding_active():
            err = store_common.sqlite_binding_error()
            hint = (f" ({err})" if err else
                    " — set RESONANT_LATTICE_DB_ENCRYPTED=1 in the launch environment "
                    "(before the plugin is imported)")
            raise RuntimeError(
                "encryption_mode=at_rest but the SQLCipher binding is not active" + hint
                + ". Install sqlcipher3 (`pip install sqlcipher3-wheels`) and export the "
                "env signal."
            )
        if not crypto_keys.kdf_available():
            raise RuntimeError(
                "encryption_mode=at_rest requires argon2-cffi (`pip install argon2-cffi`)."
            )

        eff_db_path = db_path
        if not eff_db_path:
            from hermes_constants import get_hermes_home
            eff_db_path = str(get_hermes_home() / "resonant_lattice_memory.db")
        keystore_path = self._encryption_keystore_path or (eff_db_path + ".keys")

        passphrase = crypto_keys.get_passphrase(prompt=False)
        if not passphrase:
            raise RuntimeError(
                "encryption_mode=at_rest but no passphrase available. Set "
                f"{crypto_keys.ENV_PASSPHRASE} in the environment (or run setup)."
            )
        try:
            if not os.path.exists(keystore_path):
                # First run: create the keystore. Losing the passphrase = losing the
                # data (sovereign by design); the setup UX must warn about this.
                keystore = crypto_keys.create_keystore(passphrase)
                crypto_keys.save_keystore(keystore_path, keystore)
                logger.warning(
                    "Encrypted-at-rest keystore CREATED at %s. The passphrase is the "
                    "ONLY way to decrypt this memory DB — there is NO recovery.",
                    keystore_path,
                )
            else:
                keystore = crypto_keys.load_keystore(keystore_path)
            return crypto_keys.derive_db_key(passphrase, keystore)  # verifies key-check
        except crypto_keys.WrongPassphraseError:
            raise RuntimeError(
                f"encryption passphrase does not match the keystore at {keystore_path}. "
                "Refusing to open."
            )
        finally:
            if isinstance(passphrase, bytearray):
                crypto_keys.secure_zero(passphrase)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

        # Write-gate: only "primary" agent contexts may mutate memory.
        # cron / subagent / flush contexts would otherwise ingest their own
        # system-prompt noise as user facts (see MemoryProvider ABC).
        agent_context = kwargs.get("agent_context", "primary")
        self._write_enabled = (agent_context == "primary")

        # Profile-scoped storage: prefer the HERMES_HOME the manager passes in.
        self._hermes_home = kwargs.get("hermes_home")

        try:
            from store import LatticeStore as _LatticeStore
            from retrieval import LatticeRetriever as _LatticeRetriever
        except ImportError as e:
            logger.error("CRITICAL: Failed to import Resonant Lattice Memory modules: %s", e)
            return

        db_path = None
        if self._hermes_home:
            try:
                from pathlib import Path as _Path
                db_path = str(_Path(self._hermes_home) / "resonant_lattice_memory.db")
            except Exception:
                db_path = None

        self._conflict_sim_low = float(self._config.get("conflict_sim_low", DEFAULTS["conflict_sim_low"]))
        self._conflict_sim_high = float(self._config.get("conflict_sim_high", DEFAULTS["conflict_sim_high"]))

        # Encrypted-at-rest (E0): derive the SQLCipher key (None in plaintext mode).
        # On any misconfiguration we DISABLE memory rather than persist plaintext.
        try:
            _enc_db_key = self._resolve_encryption_db_key(db_path)
        except Exception as e:
            logger.error(
                "CRITICAL: encryption setup failed (%s). Resonant Lattice Memory is "
                "DISABLED for this session.", e,
            )
            self._store = None
            self._retriever = None
            return

        try:
            self._store = _LatticeStore(
                db_path=db_path,                                   # NEW — profile-scoped
                vector_dim=self._probe_vector_dim(),
                initial_resonance=self._initial_resonance,
                decay_per_cycle=self._decay_per_cycle,
                short_tier_cycles=self._short_tier_cycles,
                mid_tier_cycles=self._mid_tier_cycles,
                promotion_threshold=self._promotion_resonance_threshold,  # maps to store's internal promotion_threshold param
                similarity_threshold=self._similarity_threshold,
                reinforce_threshold=self._reinforce_threshold,     # near-identity merge gate
                embed_model=self._embed_model,
                hrr_dim=self._hrr_dim,
                conflict_sim_low=self._conflict_sim_low,
                conflict_sim_high=self._conflict_sim_high,
                novelty_enabled=self._novelty_enabled,     # Phase 3 salience
                novelty_boost=self._novelty_boost,
                importance_categories=self._importance_categories,  # importance-weighted retention
                db_key=_enc_db_key,                        # E0: None unless encrypted
            )
        except Exception as e:
            logger.error(
                "CRITICAL: LatticeStore failed to open (%s). Resonant Lattice "
                "Memory is DISABLED for this session — lattice_store calls will "
                "report unavailable.", e, exc_info=True,
            )
            self._store = None
            self._retriever = None
            return
        mc, dc = self._store.get_cycle_counts()
        self._memory_cycle = mc
        self._dream_cycle_count = dc
        # Phase 8: remember the memory_cycle this session started at, so the
        # session-end narrative summary can stamp the cycle range it spans.
        self._session_start_cycle = mc
        # Phase 7: seed the deliberate self-model on startup (INSERT OR IGNORE —
        # never clobbers values the agent has curated since first run). Gated by
        # enable_self_model; non-fatal.
        if self._enable_self_model and self._self_model_seed:
            try:
                self._store.seed_self_model(self._self_model_seed, current_cycle=self._memory_cycle)
            except Exception as e:
                logger.debug("Self-model seeding failed (non-fatal): %s", e)
        self._retriever = _LatticeRetriever(
            self._store, self._ollama_endpoint_embed, self._embed_model,
            min_similarity=float(self._config.get("recall_floor", DEFAULTS["recall_floor"])),
            freshness_halflife=self._freshness_halflife_cycles,   # Phase 2 ranking nudge
            embed_timeout=self._embed_timeout,
            embed_keep_alive=self._embed_keep_alive,
        )
        # Tier-1 blind (collaborator): when encryption_mode=blind, BlindTier owns the HE
        # recall/HRR/maint clients, the BlindWriters, and the AEAD entity store, plus the reconcile
        # pass. The provider holds it as ONE optional field (self._blind_tier) and decorates the
        # retriever through it. Non-fatal — if the blind tier can't come up, decorate_retriever
        # returns the plaintext retriever and reconcile() is a no-op, so memory keeps working on
        # plaintext recall rather than disabling. SEAM: the entire blind-vs-plaintext divergence
        # lives behind self._blind_tier; the cognition layer never sees it (see blind_tier.py).
        self._blind_tier = None
        if self._encryption_mode == "blind":
            from blind_tier import BlindTier as _BlindTier
            eff_db_path = db_path
            if not eff_db_path:
                from hermes_constants import get_hermes_home
                eff_db_path = str(get_hermes_home() / "resonant_lattice_memory.db")
            self._blind_tier = _BlindTier.resolve(
                self._store, db_path=eff_db_path,
                keystore_path=self._encryption_keystore_path or (eff_db_path + ".keys"),
                he_keystore_path=self._blind_he_keystore_path or (eff_db_path + ".he"),
                hrr_dim=int(self._hrr_dim),
                reconcile_batch=int(self._blind_reconcile_batch),
            )
            if self._blind_tier is not None:
                self._retriever = self._blind_tier.decorate_retriever(
                    self._retriever, self._ollama_endpoint_embed, self._embed_model,
                    float(self._config.get("recall_floor", DEFAULTS["recall_floor"])))
        # P3e tool-grounding seed: ingest durable procedural/guardrail facts so the agent is grounded
        # from day one (the retriever now exists to embed them). Idempotent + non-fatal; gated on
        # write access + a non-empty seed. Embeds via the live retriever (Ollama); skipped if down.
        if self._write_enabled and self._procedural_seed:
            try:
                seed_items = []
                for content in self._procedural_seed:
                    emb = self._retriever._get_embedding(content)
                    if emb:
                        seed_items.append({"content": content, "embedding": emb})
                if seed_items:
                    seeded = self._store.seed_procedural_facts(seed_items, current_cycle=self._memory_cycle)
                    if seeded:
                        logger.info("Seeded %d procedural guardrail fact(s) for tool grounding.", seeded)
            except Exception as e:
                logger.debug("Procedural seeding failed (non-fatal): %s", e)
        from store import _ENTITY_EXTRACTOR_AVAILABLE
        _extractor_mode = "enhanced (spaCy+regex)" if _ENTITY_EXTRACTOR_AVAILABLE else "legacy (regex)"
        _hrr_mode = "rich (positional+bigram+BoW)" if _HRR_AVAILABLE else "disabled"
        logger.info(
            "\U0001f680 resonant_lattice activated \u2014 session=%s | writes=%s | "
            "Entity extraction: %s | HRR encoding: %s (dim=%d) | Hebbian cycles active.",
            session_id, self._write_enabled, _extractor_mode, _hrr_mode, self._hrr_dim,
        )

    def _probe_vector_dim(self) -> int:
        """Resolve the embedding dimension robustly.

        Priority:
          1. An existing on-disk DB's stored vec dimension is authoritative —
             prevents a transient Ollama outage at startup from pinning the store
             into degraded (FTS-only) mode for the whole session.
          2. Otherwise probe Ollama (with retries).
          3. Last resort: 768.
        """
        # 1. Trust an existing DB's dimension over a live probe.
        try:
            from pathlib import Path as _Path
            if self._hermes_home:
                _db = _Path(self._hermes_home) / "resonant_lattice_memory.db"
                if _db.exists():
                    dim = self._read_db_vector_dim(str(_db))
                    if dim:
                        return dim
        except Exception:
            pass

        # 2. Probe Ollama with a couple of retries.
        import time as _time
        for _attempt in range(3):
            if _attempt:
                _time.sleep(1.5 * _attempt)
            try:
                payload = {"model": self._embed_model, "prompt": "probe"}
                if self._embed_keep_alive:
                    payload["keep_alive"] = self._embed_keep_alive
                req = urllib.request.Request(
                    f"{self._ollama_endpoint_embed}/api/embeddings",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self._embed_timeout) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    dim = len(res.get("embedding", []))
                    if dim:
                        return dim
            except Exception:
                continue

        # 3. Fallback.
        logger.warning(
            "resonant_lattice: could not determine embedding dim (no existing DB, "
            "Ollama probe failed) — defaulting to 768. Entering DEGRADED (FTS-only) mode "
            "until embed succeeds or DB is rebuilt. This is normal on first run or cold embedder."
        )
        return 768

    @staticmethod
    def _read_db_vector_dim(db_path: str) -> Optional[int]:
        """Read the stored semantic_vec dimension from an existing DB (no deps)."""
        try:
            try:
                import pysqlite3 as _sqlite3
            except ImportError:
                import sqlite3 as _sqlite3
            conn = _sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type='table' AND name='semantic_vec'"
                ).fetchone()
            finally:
                conn.close()
            if row and row[0]:
                m = re.search(r'float\[(\d+)\]', row[0])
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        return None

    def get_feature_status(self) -> dict:
        """Return which optional roadmap phases are currently active.
        Useful for introspection, memory_audit, and docs."""
        return {
            "core_features": True,  # 3-tier, HRR, entities, P1 temporal, P2 freshness, P3 salience, P6 conflicts, etc.
            "p4_gist_before_prune": bool(getattr(self, "_gist_before_prune", False)),
            "p5_relations": bool(getattr(self, "_enable_relations", False)),
            "p7_self_model": bool(getattr(self, "_enable_self_model", False)),
            "p8_narrative": bool(getattr(self, "_enable_narrative", False)),
            "conflict_limbo": bool(getattr(self, "_conflict_limbo", True)),
            "tool_memory": bool(getattr(self, "_enable_tool_memory", True)),
            "gate_self_writes": bool(getattr(self, "_gate_self_writes", True)),
        }

    def get_defaults(self) -> dict:
        """Return the centralized defaults (from config_schema.DEFAULTS).
        Useful for introspection and tooling."""
        from config_schema import DEFAULTS
        return dict(DEFAULTS)  # copy

    def system_prompt_block(self) -> str:
        if not self._store:
            return "# Resonant Lattice Memory — Initializing..."
        block = (
            "# Resonant Lattice Memory Active\n"
            "Neuroplastic Hebbian system with 3-tier resonance (short/mid/long), "
            "HRR compositional algebra, entity graph, and cycle-driven Dream Cycles.\n"
            "\n"
            "Recalled memory is a set of FALLIBLE RETRIEVED CANDIDATES, not ground "
            "truth. Treat anything inside <resonant_memory> as a hint that may be "
            "approximate, outdated, or a semantically-similar near-miss — never as a "
            "verbatim quote.\n"
            "- An empty or weak recall is a VALID, SUCCESSFUL outcome. If memory "
            "does not contain something, say \"I don't have that in memory\" rather "
            "than guessing or inventing a fact.\n"
            "- NEVER present a reconstruction or a similar neighbour as a verbatim "
            "stored fact. To assert the exact content of a specific stored row, call "
            "lattice_store get_fact with its ID; a found:false result means it is "
            "NOT stored — report that, do not substitute a neighbour.\n"
            "- Each candidate carries [ID | Tier | Res] metadata: low Res or a "
            "'short' tier means weak/uncertain (more likely stale or noisy). A "
            "'long' tier with high Res is more reliable, but still confirm exact "
            "wording with get_fact before quoting.\n"
            "- A [CONFLICT LOCK] tag (or an 'unresolved' note) means memory holds "
            "contradictory facts about something. The system duels them automatically, "
            "but when you know which is right you may call lattice_store "
            "pending_conflicts to see the competitors and resolve_conflict (fact_id = "
            "the correct one) to settle it — never silently pick one as truth.\n"
            "Use lattice_store for manual control and feedback."
        )
        # Phase 7: surface the curated self-model DETERMINISTICALLY (not via fallible
        # fuzzy recall). Read-only here — the ingest paths never write this store, so
        # it is authoritative about the agent itself. Updated only via set_self_model.
        if self._enable_self_model:
            try:
                identity = self._store.get_self_model()
            except Exception:
                identity = None
            if identity:
                lines = "\n".join(f"- {r['key']}: {r['value']}" for r in identity)
                block += (
                    "\n\n# Agent Self-Model (curated, authoritative)\n"
                    "This is your deliberately maintained identity — unlike recalled "
                    "memory it is authoritative about yourself and is never auto-ingested. "
                    "Update it only via lattice_store set_self_model.\n" + lines
                )
        # Phase 8: surface recent cross-session narrative as ambient "recent history"
        # continuity. Explicitly framed as summary (not verbatim). Most recent few
        # only; the full bounded log is available via lattice_store narrative.
        if self._enable_narrative:
            try:
                recent = self._store.get_recent_narrative(limit=self._narrative_surface)
            except Exception:
                recent = None
            if recent:
                lines = "\n".join(f"- {r['summary']}" for r in recent)
                block += (
                    "\n\n# Recent History (across sessions)\n"
                    "Summaries of what you and the user have been doing together "
                    "(remembered gist, not verbatim — for continuity, not quoting). "
                    "Use lattice_store narrative for more.\n" + lines
                )
        return block

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        if not self._store:
            return
        if not self._write_enabled:
            return

        self._turn_count += 1
        sid = session_id or self._session_id
        turn_at_spawn = self._turn_count
        prev_thread = self._last_ingest_thread   # ordering anchor for this turn

        def _ingest():
            try:
                # Preserve cross-turn episode ordering: wait (bounded) for the
                # previous turn's ingest to land its rows before inserting ours.
                # Without this, a descheduled or consolidation-stalled predecessor
                # lets this turn's episodes commit first, scrambling the
                # transcript the consolidation LLM reads (it orders by id).
                if prev_thread is not None and prev_thread.is_alive():
                    prev_thread.join(timeout=30.0)

                self._store.add_turn(sid, user_content, assistant_content)

                if turn_at_spawn % self._reflection_frequency == 0:
                    self._run_consolidation_epoch(sid)

                # === First-class Tool & Action Memory ===
                if self._enable_tool_memory and messages:
                    try:
                        actions = self._extract_tool_actions(messages)
                        for action in actions:
                            self._ingest_tool_action(action, sid)
                    except Exception as e:
                        logger.debug("Tool memory ingestion failed (non-fatal): %s", e)

            except Exception as e:
                logger.error("sync_turn ingest failed: %s", e)

        t = threading.Thread(target=_ingest, daemon=True)
        t.start()
        self._last_ingest_thread = t

    def _is_self_referential_infra(self, content: str) -> bool:
        """Policy gate (Phase E): True if `content` describes the AGENT'S OWN
        configuration / infrastructure / identity rather than the user or domain.

        Deliberate POLICY BOUNDARY, not a cleverness heuristic. Its only job is to
        stop the memory system from autonomously persisting its own operational
        chatter — model name, context size, IP/endpoint, training/identity — as if
        it were a durable user fact (which pollutes the user model and inflates
        self-referential resonance).

        Conservative two-tier rule over explicit, auditable lists:
          1) any _SELF_INFRA_PHRASES substring is unambiguous on its own, OR
          2) an AI-self _SELF_SUBJECTS term co-occurs with an _INFRA_TERMS
             descriptor (neither fires alone).
        It UNDER-blocks on purpose — it will miss a novel paraphrase rather than
        risk dropping a legitimate fact about the USER's infrastructure or the
        user's project.

        Delegates to the pure self_write_gate.is_self_referential_infra; the
        denylists live there. Kept as a method so call sites (consolidation,
        on_memory_write) stay self._is_self_referential_infra(content).
        """
        return is_self_referential_infra(content)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Fields for `hermes memory setup`. All local — no secrets/credentials.

        NOTE on promotion latency: tier dwell is counted in DREAM CYCLES, and a
        dream cycle fires every `dream_every_n_consolidations` consolidations
        (each ~`reflection_frequency` turns). So short→mid ≈
        short_tier_cycles × dream_every_n_consolidations × reflection_frequency
        turns. Lower the *_tier_cycles values for faster promotion.

        The field list itself lives in config_schema.CONFIG_SCHEMA (text/data only).
        """
        return CONFIG_SCHEMA

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist non-secret config under plugins.resonant_lattice in config.yaml.

        Mirrors the read path in _load_plugin_config (merge, don't clobber other
        plugins/sections). All fields here are local, so nothing goes to .env.
        """
        try:
            import yaml
            from pathlib import Path as _Path
            config_path = _Path(hermes_home) / "config.yaml"
            all_config = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    all_config = yaml.safe_load(f) or {}
            all_config.setdefault("plugins", {})
            section = all_config["plugins"].setdefault("resonant_lattice", {})
            section.update(values)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(all_config, f, sort_keys=False, allow_unicode=True)
            logger.info("resonant_lattice: saved %d config value(s) to %s", len(values), config_path)
        except Exception as e:
            logger.error("resonant_lattice save_config failed: %s", e)


# ----------------------------------------------------------------------
# Plugin registration
# ----------------------------------------------------------------------
def _load_plugin_config() -> dict:
    """Load config from profile-scoped config.yaml."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import cfg_get
        import yaml
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "resonant_lattice", default={}) or {}
    except Exception:
        return {}

def register(ctx) -> None:
    try:
        config = _load_plugin_config()
        provider = LatticeMemoryProvider(config=config)
        ctx.register_memory_provider(provider)
    except Exception as e:
        logger.error(f"🔥 CRITICAL ERROR INSTANTIATING LATTICE MEMORY: {e}", exc_info=True)