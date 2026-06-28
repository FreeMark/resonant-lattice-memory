"""
store.py — Resonant Lattice Memory Neuroplastic Store

Core storage engine for the Hebbian dual-layer memory system.
Features:
- Dynamic vector dimension support (sqlite-vec)
- Hybrid HRR phase vectors (compositional algebra)
- Entity graph (fast regex extraction + fact_entities junction)
- Cycle-driven Hebbian operations: exponential decay, resonance reinforcement,
  tier promotion (short → mid → long), abstraction/generalization, and
  HRR-based conflict resolution
- Zero LLM calls during entity extraction (deterministic regex)
- All data returned to agent/prefetch is clean (no raw vectors/bytes)
- Thread-safe with RLock
"""

import threading
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import sqlite_vec   # hard dep: import fails here without it, so the store tests skip

# Shared primitives + optional-dependency imports live in store_common (a leaf
# module) so the store_* mixins can import them without a circular reference back
# to this module. serialize_vector and _ENTITY_EXTRACTOR_AVAILABLE are re-exported
# here because external consumers do `from store import serialize_vector` /
# `from store import _ENTITY_EXTRACTOR_AVAILABLE`.
from store_common import (
    serialize_vector,
    hrr,
    _HRR_AVAILABLE,
    sqlite3,
    _extract_entities_fn,
    _ENTITY_EXTRACTOR_AVAILABLE,
)

try:
    from config_schema import DEFAULTS as _STORE_DEFAULTS
except Exception:
    _STORE_DEFAULTS = {}

# Behaviour-preserving structural split: LatticeStore's methods live in sibling
# mixins, composed here. Mixins use FLAT sibling imports (never package-relative)
# and never import LatticeStore — they rely on attributes the composite defines.
from store_schema import SchemaMixin
from store_facts import FactsMixin
from store_dream import DreamCycleMixin
from store_abstraction import AbstractionMixin
from store_episodes import EpisodesMixin
from store_entities import EntitiesMixin
from store_relations import RelationsMixin
from store_identity import IdentityMixin
from store_narrative import NarrativeMixin
from store_blind import BlindMixin

logger = logging.getLogger(__name__)


class LatticeStore(SchemaMixin, FactsMixin, DreamCycleMixin, AbstractionMixin,
                   EpisodesMixin, EntitiesMixin, RelationsMixin, IdentityMixin,
                   NarrativeMixin, BlindMixin):
 
    def __init__(
        self,
        db_path: "str | Path | None" = None,
        vector_dim: int = 768,
        initial_resonance: int = _STORE_DEFAULTS.get("initial_resonance", 4),
        decay_per_cycle: float = _STORE_DEFAULTS.get("decay_per_cycle", 0.5),
        short_tier_cycles: int = _STORE_DEFAULTS.get("short_tier_cycles", 3),
        mid_tier_cycles: int = _STORE_DEFAULTS.get("mid_tier_cycles", 6),
        promotion_threshold: int = _STORE_DEFAULTS.get("promotion_resonance_threshold", 4),
        similarity_threshold: float = _STORE_DEFAULTS.get("similarity_threshold", 0.78),
        reinforce_threshold: float = _STORE_DEFAULTS.get("reinforce_threshold", 0.95),
        embed_model: str = _STORE_DEFAULTS.get("embed_model", "nomic-embed-text"),
        hrr_dim: int = _STORE_DEFAULTS.get("hrr_dim", 1024),
        conflict_sim_low: float = _STORE_DEFAULTS.get("conflict_sim_low", 0.55),
        conflict_sim_high: float = _STORE_DEFAULTS.get("conflict_sim_high", 0.90),
        novelty_enabled: bool = _STORE_DEFAULTS.get("novelty_enabled", True),
        novelty_boost: float = _STORE_DEFAULTS.get("novelty_boost", 2.0),
        detect_policy_conflicts: bool = _STORE_DEFAULTS.get("detect_policy_conflicts", True),
        importance_categories=None,
        db_key: "bytes | bytearray | None" = None,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "resonant_lattice_memory.db")
 
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
 
        self.vector_dim = vector_dim
        self.conflict_sim_low = float(conflict_sim_low)
        self.conflict_sim_high = float(conflict_sim_high)
        # Phase 3 salience: a novel (low top-similarity) fact enters at higher
        # resonance so an important one-shot ("your daughter is named Maya")
        # survives without repetition, while a near-duplicate gets little boost.
        self.novelty_enabled = bool(novelty_enabled)
        self.novelty_boost = float(novelty_boost)
        self.detect_policy_conflicts = bool(detect_policy_conflicts)
        # Importance-weighted retention: facts in these categories decay slower
        # (apply_cycle_decay importance_discount). Default from central config.
        _imp = importance_categories if importance_categories is not None \
            else _STORE_DEFAULTS.get("importance_categories", [])
        self.importance_categories = {str(c).strip().lower() for c in _imp if str(c).strip()}
        self.initial_resonance = initial_resonance
        self.decay_per_cycle = decay_per_cycle
        self.short_tier_cycles = short_tier_cycles
        self.mid_tier_cycles = mid_tier_cycles
        self.promotion_threshold = promotion_threshold
        # NOTE: These defaults are kept in sync with LatticeMemoryProvider and
        # recommended_config.yaml. The provider always overrides them with
        # config-aware values (including promotion_resonance_threshold).
        # INVARIANT: a fact must be able to reach the promotion bar. If the
        # initial resonance starts below promotion_threshold, an un-reinforced
        # fact decays away from the bar and can never promote (it just lingers
        # in 'short' indefinitely). Warn loudly so this is a deliberate choice,
        # not a silent dead tier.
        if self.initial_resonance < self.promotion_threshold:
            logger.warning(
                "initial_resonance (%.2f) < promotion_threshold (%d): facts that "
                "are never reinforced/recalled can never promote out of 'short'. "
                "This is intended only if you want recall to be REQUIRED for "
                "long-term retention. Otherwise raise initial_resonance.",
                self.initial_resonance, self.promotion_threshold,
            )
        elif self.initial_resonance == self.promotion_threshold and self.decay_per_cycle > 0:
            logger.info(
                "initial_resonance == promotion_threshold (%.2f) with decay active: "
                "the first dream-cycle decay drops un-recalled facts below the "
                "promotion bar before tier dwell is satisfied, so recall/feedback "
                "is effectively REQUIRED for promotion. Raise initial_resonance "
                "above the threshold if passive promotion is desired.",
                float(self.promotion_threshold),
            )
        self.similarity_threshold = similarity_threshold   # recall/search/dedup cutoff
        # Silent-merge gate: only fold a NEW fact into an existing one at
        # near-identity. The 0.78–0.95 band (close but not identical) often holds
        # contradictory updates ("prefers dark" vs "prefers light"), which must be
        # stored separately so conflict detection can see them — not dropped.
        self.reinforce_threshold = max(float(reinforce_threshold), float(similarity_threshold))
        self.embed_model = embed_model
        self.hrr_dim = int(hrr_dim)                        # HRR phase-vector dim (distinct from embedding vector_dim)
        self.degraded = False                              # set True on embedding-dim mismatch (observable via stats)
 
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10.0
        )
        # Encrypted-at-rest (E0): unlock the SQLCipher DB with the raw key as the
        # VERY FIRST statement, before any other SQL. The binding is sqlcipher3
        # (selected in store_common via RESONANT_LATTICE_DB_ENCRYPTED). We take
        # ownership of the key buffer and wipe it once SQLCipher holds it — PRAGMA
        # key takes no bound parameter, so the hex value is formatted in (hex only,
        # no injection surface). db_key is None in the plaintext path (no change).
        if db_key is not None:
            import crypto_keys
            try:
                self._conn.execute(
                    'PRAGMA key = "%s"' % crypto_keys.db_key_to_pragma_value(db_key)
                )
            finally:
                if isinstance(db_key, bytearray):
                    crypto_keys.secure_zero(db_key)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        # Enable foreign-key enforcement so ON DELETE CASCADE / SET NULL in
        # abstraction_sources actually fire. Without this, pruning a source fact
        # leaves dangling provenance rows and the staleness signal under-reports
        # lost evidence. Must be set per-connection, before any DML.
        try:
            self._conn.execute("PRAGMA foreign_keys=ON")
        except Exception as e:
            logger.warning("Could not enable foreign_keys pragma: %s", e)
 
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except Exception as e:
            logger.error("sqlite-vec failed to load: %s", e)
            raise
 
        self._validate_vector_dim()   # NEW — detect dim mismatch on reopen
        self._init_db()
        self._migrate_schema()         # NEW — ALTER-add cycles_in_tier; cosine-rebuild vec table
        self._stamp_meta()             # NEW — record/validate hrr_dim + encoding version
        self._load_or_init_cycle_counters()   # NEW — persist memory/dream cycle counters across restarts
                
    def _load_or_init_cycle_counters(self) -> None:
        """Ensure memory_cycle and dream_cycle exist in meta (defaults to 0 on first run)."""
        with self._lock:
            for key, default in [("memory_cycle", 0), ("dream_cycle", 0)]:
                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key=?", (key,)
                ).fetchone()
                if not row:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                        (key, str(default))
                    )
            self._conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate counts safe for serialisation to JSON."""
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM semantic_facts"
            ).fetchone()[0]
 
            by_tier = {
                row["tier"]: row["cnt"]
                for row in self._conn.execute(
                    "SELECT tier, COUNT(*) as cnt FROM semantic_facts GROUP BY tier"
                ).fetchall()
            }
 
            episodes = self._conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]

            entities = self._conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]

            return {
                "total_facts": total,
                "by_tier": by_tier,
                "total_episodes": episodes,
                "total_entities": entities,
                "degraded": self.degraded,        # True ⇒ embedding-dim mismatch, FTS-only
                "vector_dim": self.vector_dim,
                "hrr_dim": self.hrr_dim,
            }

    def get_memory_health(self, near_cap: float = 49.0) -> Dict[str, Any]:
        """Read-only memory-health snapshot for the cycle-based audit / memory_audit
        tool action. No side effects.

        Cheap aggregate queries only (plus one GROUP BY over abstraction
        provenance), so it is safe to run every dream cycle or on demand. Surfaces
        the things that silently drift on a long-lived store: tier growth,
        resonance saturation, active conflicts, entity/orphan counts, procedural
        backlog, and abstractions that have lost all supporting evidence.
        """
        with self._lock:
            by_tier = {
                r["tier"]: r["cnt"] for r in self._conn.execute(
                    "SELECT tier, COUNT(*) AS cnt FROM semantic_facts GROUP BY tier"
                ).fetchall()
            }
            by_category = {
                r["category"]: r["cnt"] for r in self._conn.execute(
                    "SELECT category, COUNT(*) AS cnt FROM semantic_facts GROUP BY category"
                ).fetchall()
            }
            near_cap_facts = self._conn.execute(
                "SELECT COUNT(*) FROM semantic_facts WHERE resonance_count >= ?",
                (near_cap,),
            ).fetchone()[0]
            conflict_groups = self._conn.execute(
                "SELECT COUNT(DISTINCT conflict_group_id) FROM semantic_facts "
                "WHERE conflict_group_id IS NOT NULL"
            ).fetchone()[0]
            conflicted_facts = self._conn.execute(
                "SELECT COUNT(*) FROM semantic_facts WHERE conflict_group_id IS NOT NULL"
            ).fetchone()[0]
            total_entities = self._conn.execute(
                "SELECT COUNT(*) FROM entities"
            ).fetchone()[0]
            orphan_entities = self._conn.execute(
                "SELECT COUNT(*) FROM entities "
                "WHERE entity_id NOT IN (SELECT DISTINCT entity_id FROM fact_entities)"
            ).fetchone()[0]
            episodes = self._conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]
            tool_total = self._conn.execute(
                "SELECT COUNT(*) FROM tool_episodes"
            ).fetchone()[0]
            tool_undistilled = self._conn.execute(
                "SELECT COUNT(*) FROM tool_episodes WHERE distilled = 0"
            ).fetchone()[0]
            # Abstraction provenance health: count abstractions that have lost ALL
            # supporting evidence (the staleness == 1.0 case) via one GROUP BY —
            # a cheap proxy for the high-staleness signal, no per-abstraction
            # recomputation.
            prov = self._conn.execute(
                """
                SELECT s.abstract_id AS aid,
                       COUNT(*) AS total_sources,
                       SUM(CASE WHEN f.id IS NOT NULL AND f.resonance_count > 0
                                THEN 1 ELSE 0 END) AS active_sources
                FROM abstraction_sources s
                LEFT JOIN semantic_facts f ON f.id = s.source_id
                GROUP BY s.abstract_id
                """
            ).fetchall()
            abstractions_tracked = len(prov)
            abstractions_evidence_gone = sum(
                1 for r in prov
                if (r["total_sources"] or 0) > 0 and (r["active_sources"] or 0) == 0
            )
            return {
                "total_facts": sum(by_tier.values()),
                "by_tier": by_tier,
                "by_category": by_category,
                "long_tier_facts": by_tier.get("long", 0),
                "near_cap_facts": near_cap_facts,
                "near_cap_threshold": near_cap,
                "active_conflict_groups": conflict_groups,
                "conflicted_facts": conflicted_facts,
                "total_entities": total_entities,
                "orphan_entities": orphan_entities,
                "total_episodes": episodes,
                "tool_episodes_total": tool_total,
                "tool_episodes_undistilled": tool_undistilled,
                "abstractions_tracked": abstractions_tracked,
                "abstractions_evidence_gone": abstractions_evidence_gone,
                "degraded": self.degraded,
                "vector_dim": self.vector_dim,
                "hrr_dim": self.hrr_dim,
            }

    def get_cycle_counts(self) -> tuple[int, int]:
        """Return (memory_cycle, dream_cycle) persisted in meta table."""
        with self._lock:
            mc = self._conn.execute(
                "SELECT value FROM meta WHERE key='memory_cycle'"
            ).fetchone()
            dc = self._conn.execute(
                "SELECT value FROM meta WHERE key='dream_cycle'"
            ).fetchone()
            return (
                int(mc["value"]) if mc else 0,
                int(dc["value"]) if dc else 0
            )

    def set_cycle_counts(self, memory_cycle: Optional[int] = None,
                         dream_cycle: Optional[int] = None) -> None:
        """Persist cycle counters — pass ONLY the counter you changed.

        Consolidation epochs and dream cycles run on different threads under
        different locks. Writing both keys from both paths let a slow writer
        overwrite the other path's counter with a stale value. None = leave
        that key untouched. Calling with both still works (seeding, tests).
        """
        with self._lock:
            if memory_cycle is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('memory_cycle', ?)",
                    (str(memory_cycle),)
                )
            if dream_cycle is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('dream_cycle', ?)",
                    (str(dream_cycle),)
                )
            self._conn.commit()

    def close(self) -> None:
        """Close the DB connection. Lock-guarded so in-flight locked operations
        (background ingest/dream threads) drain before the handle dies.
        Idempotent: safe to call twice."""
        with self._lock:
            try:
                self._conn.close()
            except Exception as e:
                logger.debug("LatticeStore close: %s", e)