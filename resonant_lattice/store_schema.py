"""store_schema.py — SchemaMixin: DB creation, idempotent migrations, meta.

Mixed into LatticeStore. Relies on the composite for self._conn, self._lock,
self.vector_dim, self.hrr_dim, self.degraded, self.db_path."""

import logging
import re

logger = logging.getLogger(__name__)


class SchemaMixin:
        
    def _validate_vector_dim(self) -> None:
        """Warn if the on-disk vec0 dimension differs from configured dim."""
        try:
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_vec'"
            ).fetchone()
            if row and row["sql"]:
                m = re.search(r'float\[(\d+)\]', row["sql"])
                if m:
                    stored_dim = int(m.group(1))
                    if stored_dim != self.vector_dim:
                        self.degraded = True
                        logger.warning(
                            "DIMENSION MISMATCH: DB has float[%d] but configured dim=%d. "
                            "Entering DEGRADED (FTS-only) mode. Embeddings will use FTS5 fallback only. "
                            "Either delete the DB to rebuild, or set vector_dim=%d in your config.",
                            stored_dim, self.vector_dim, stored_dim
                        )
        except Exception:
            pass  # non-fatal — best-effort validation only


    def _init_db(self) -> None:
        """Create tables with HRR + Entity Graph support."""
        schema = f"""
        PRAGMA journal_mode=WAL;

        -- Episodic layer (L1)
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

        -- Neuroplastic Semantic Core (L2)
        CREATE TABLE IF NOT EXISTS semantic_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tier TEXT DEFAULT 'short',
            resonance_count REAL DEFAULT 3.0,
            cycles_in_tier INTEGER DEFAULT 0,
            source_session TEXT,
            hrr_vector BLOB,
            conflict_group_id TEXT,
            source_quote TEXT,
            source_ref TEXT,
            quote_status TEXT,
            learned_at_cycle INTEGER,
            last_confirmed_cycle INTEGER,
            superseded_by INTEGER REFERENCES semantic_facts(id) ON DELETE SET NULL,
            superseded_at_cycle INTEGER,
            max_resonance_seen REAL,
            conflict_since_cycle INTEGER,
            pinned INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Entity Graph
        CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS fact_entities (
            fact_id INTEGER REFERENCES semantic_facts(id),
            entity_id INTEGER REFERENCES entities(entity_id),
            PRIMARY KEY (fact_id, entity_id)
        );

        -- Abstraction provenance (back-pointers from synthesized facts to source facts)
        CREATE TABLE IF NOT EXISTS abstraction_sources (
            link_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            abstract_id INTEGER REFERENCES semantic_facts(id) ON DELETE CASCADE,
            source_id   INTEGER REFERENCES semantic_facts(id) ON DELETE SET NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cluster_size_at_creation INTEGER,
            UNIQUE (abstract_id, source_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_semantic_facts_conflict ON semantic_facts(conflict_group_id);

        -- Phase 1: supersedion chain lookups (partial — only superseded rows)
        CREATE INDEX IF NOT EXISTS idx_semantic_facts_superseded
            ON semantic_facts(superseded_by) WHERE superseded_by IS NOT NULL;

        -- Hebbian hot-path indexes (promotion, decay, conflict, stats)
        CREATE INDEX IF NOT EXISTS idx_semantic_facts_tier_resonance_dwell
            ON semantic_facts(tier, resonance_count, cycles_in_tier);
        CREATE INDEX IF NOT EXISTS idx_semantic_facts_conflict_active
            ON semantic_facts(conflict_group_id) WHERE conflict_group_id IS NOT NULL;

        -- Key/value metadata (encoding version, hrr_dim, vec metric)
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        -- Vector index (dynamic dimension, cosine distance)
        CREATE VIRTUAL TABLE IF NOT EXISTS semantic_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{self.vector_dim}] distance_metric=cosine
        );

        -- FTS5 fallback
        CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
            content, category, content=semantic_facts, content_rowid=id
        );

        -- Triggers for FTS5 + vector cleanup
        CREATE TRIGGER IF NOT EXISTS semantic_facts_ai AFTER INSERT ON semantic_facts BEGIN
            INSERT INTO semantic_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
        END;

        CREATE TRIGGER IF NOT EXISTS semantic_facts_ad AFTER DELETE ON semantic_facts BEGIN
            INSERT INTO semantic_fts(semantic_fts, rowid, content, category)
                VALUES ('delete', old.id, old.content, old.category);
            DELETE FROM semantic_vec WHERE id = old.id;
            DELETE FROM fact_entities WHERE fact_id = old.id;
        END;

        CREATE TRIGGER IF NOT EXISTS semantic_facts_au AFTER UPDATE ON semantic_facts
        WHEN old.content IS NOT new.content OR old.category IS NOT new.category BEGIN
            INSERT INTO semantic_fts(semantic_fts, rowid, content, category)
                VALUES ('delete', old.id, old.content, old.category);
            INSERT INTO semantic_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
        END;
        """
        with self._lock:
            self._conn.executescript(schema)
            self._conn.commit()
        logger.info("LatticeStore schema initialized (HRR + Entity Graph ready).")


    # ====================== SCHEMA MIGRATION ======================
    def _migrate_schema(self) -> None:
        """Run idempotent migrations for existing databases.

        Thin orchestrator — each step is a named helper, safe to run on every
        open (they self-detect whether work is needed).
        """
        self._migrate_add_columns()
        self._migrate_add_source_provenance()
        self._migrate_add_temporal()
        self._migrate_add_salience()
        self._migrate_add_conflict_since()
        self._migrate_add_entities_dirty()
        self._migrate_add_dormant_since()
        self._migrate_add_pinned()
        self._migrate_add_relations()
        self._migrate_add_agent_identity()
        self._migrate_add_session_summaries()
        self._migrate_add_semantic_he()
        self._migrate_add_semantic_he_hrr()
        self._migrate_add_semantic_he_meta()
        self._migrate_add_semantic_he_entities()
        self._migrate_add_reencrypt_audit()
        self._migrate_vec_to_cosine()
        self._migrate_fts_trigger()
        self._migrate_delete_trigger()
        self._migrate_add_abstraction_sources()
        self._migrate_add_tool_episodes()
        self._migrate_add_canonical_facts()
        self._migrate_add_write_batches()


    def _migrate_add_columns(self) -> None:
        """Add semantic_facts.cycles_in_tier if missing.

        Turn/cycle-based tier-dwell counter — no wall-clock involved. Existing
        rows default to 0.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "cycles_in_tier" not in cols:
                    self._conn.execute(
                        "ALTER TABLE semantic_facts ADD COLUMN cycles_in_tier INTEGER DEFAULT 0"
                    )
                    self._conn.commit()
                    logger.info("Migration: added semantic_facts.cycles_in_tier")
            except Exception as e:
                logger.error("Migration (cycles_in_tier) failed: %s", e)


    def _migrate_add_source_provenance(self) -> None:
        """Add semantic_facts.source_quote / source_ref / quote_status if missing.

        Grounded-extraction provenance (Phase D + attestation): source_quote holds
        the verbatim snippet a fact was derived from, source_ref an optional
        URL/identifier when the turn came from a tool/web fetch, and quote_status
        the verification verdict ('attested' / 'soft' / 'unattested' / 'unverified'
        / NULL). All nullable TEXT; legacy rows default to NULL. Self-detecting +
        idempotent — mirrors _migrate_add_columns, safe to run on every open. No
        wall-clock involved.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                for col in ("source_quote", "source_ref", "quote_status"):
                    if col not in cols:
                        self._conn.execute(
                            f"ALTER TABLE semantic_facts ADD COLUMN {col} TEXT"
                        )
                        logger.info("Migration: added semantic_facts.%s", col)
                self._conn.commit()
            except Exception as e:
                logger.error("Migration (source provenance) failed: %s", e)


    def _migrate_add_temporal(self) -> None:
        """Add the Phase-1 temporal/supersedion columns to semantic_facts.

        learned_at_cycle / last_confirmed_cycle stamp the memory_cycle at first
        INSERT and at last reinforcement (the logical clock — never wall-clock).
        superseded_by / superseded_at_cycle record a conflict-loser's replacement
        and are populated in Phase 1b; superseded_by is a self-FK with ON DELETE
        SET NULL so a pruned winner doesn't leave a dangling pointer (SQLite
        permits a REFERENCES column via ALTER only when its default is NULL — it
        is). All nullable INTEGER; legacy rows stay NULL and are back-stamped
        lazily by ordinary reinforcement (we never fabricate a learned_at for a
        pre-existing row). Self-detecting + idempotent like the sibling column
        migrations — safe to run on every open. No wall-clock involved.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                col_defs = {
                    "learned_at_cycle": "INTEGER",
                    "last_confirmed_cycle": "INTEGER",
                    "superseded_by": "INTEGER REFERENCES semantic_facts(id) ON DELETE SET NULL",
                    "superseded_at_cycle": "INTEGER",
                }
                for col, decl in col_defs.items():
                    if col not in cols:
                        self._conn.execute(
                            f"ALTER TABLE semantic_facts ADD COLUMN {col} {decl}"
                        )
                        logger.info("Migration: added semantic_facts.%s", col)
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_semantic_facts_superseded "
                    "ON semantic_facts(superseded_by) WHERE superseded_by IS NOT NULL"
                )
                self._conn.commit()
            except Exception as e:
                logger.error("Migration (temporal) failed: %s", e)


    def _migrate_add_salience(self) -> None:
        """Add semantic_facts.max_resonance_seen (Phase 3) if missing.

        A high-water mark of a fact's resonance — 'was this ever important' — used
        by Phase 4 to decide whether a fading fact earned a gist before pruning.
        Nullable REAL. On legacy DBs we backfill it to the current resonance_count
        (a safe lower-bound estimate of the peak) so pre-existing facts aren't all
        treated as never-important. Self-detecting + idempotent like the sibling
        column migrations. No wall-clock involved.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "max_resonance_seen" not in cols:
                    self._conn.execute(
                        "ALTER TABLE semantic_facts ADD COLUMN max_resonance_seen REAL"
                    )
                    logger.info("Migration: added semantic_facts.max_resonance_seen")
                # Backfill NULLs to the current resonance (idempotent: only touches
                # rows that still lack a peak — fresh inserts set it themselves).
                self._conn.execute(
                    "UPDATE semantic_facts SET max_resonance_seen = resonance_count "
                    "WHERE max_resonance_seen IS NULL"
                )
                self._conn.commit()
            except Exception as e:
                logger.error("Migration (salience) failed: %s", e)


    def _migrate_add_conflict_since(self) -> None:
        """Add semantic_facts.conflict_since_cycle (Phase 6) if missing.

        The memory_cycle at which a fact entered its current conflict group, so the
        'conflicts as conversation' surface can wait until a conflict has persisted
        a few cycles (don't nag the instant a duel starts) and report a group's age.
        Nullable INTEGER; set when resolve_hrr_conflicts groups a pair, cleared when
        the conflict resolves. Self-detecting + idempotent; no wall-clock.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "conflict_since_cycle" not in cols:
                    self._conn.execute(
                        "ALTER TABLE semantic_facts ADD COLUMN conflict_since_cycle INTEGER"
                    )
                    self._conn.commit()
                    logger.info("Migration: added semantic_facts.conflict_since_cycle")
            except Exception as e:
                logger.error("Migration (conflict_since_cycle) failed: %s", e)


    def _migrate_add_entities_dirty(self) -> None:
        """Add semantic_facts.entities_dirty (blind-tier entity re-mirror flag) if missing.

        The AEAD entity set in semantic_he_entities is the ONE blind mirror whose source is
        MUTABLE — reinforcement links new entities to an existing fact (store_facts._link_entities),
        so 'mirror once when the row is missing' goes stale. This 0/1 flag is set when a genuinely
        new entity link lands and cleared once the set is re-mirrored, so facts_needing_entity_mirror
        picks the change up. NOT NULL INTEGER default 0; legacy rows default 0 (a clean fact whose
        existing mirror, if any, is current as of its last link — the next real link flips it).
        Self-detecting + idempotent like the sibling column migrations. No wall-clock involved.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "entities_dirty" not in cols:
                    self._conn.execute(
                        "ALTER TABLE semantic_facts ADD COLUMN entities_dirty INTEGER NOT NULL DEFAULT 0"
                    )
                    self._conn.commit()
                    logger.info("Migration: added semantic_facts.entities_dirty")
            except Exception as e:
                logger.error("Migration (entities_dirty) failed: %s", e)


    def _migrate_add_dormant_since(self) -> None:
        """Add semantic_facts.dormant_since_cycle (buried-but-pluckable forget policy, P2b) if missing.

        The memory_cycle at which a fully-faded fact (resonance <= 0) first became DORMANT. Under the
        'demote then deep-delete' policy, a dormant fact is kept (still pluckable by a strong cue) and
        only truly deleted after it has stayed dormant for the configured grace (cycle-driven, no
        wall-clock). Stamped/cleared by prune_weak_facts; nullable INTEGER. Self-detecting +
        idempotent like the sibling column migrations.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "dormant_since_cycle" not in cols:
                    self._conn.execute(
                        "ALTER TABLE semantic_facts ADD COLUMN dormant_since_cycle INTEGER"
                    )
                    self._conn.commit()
                    logger.info("Migration: added semantic_facts.dormant_since_cycle")
            except Exception as e:
                logger.error("Migration (dormant_since_cycle) failed: %s", e)


    def _migrate_add_pinned(self) -> None:
        """Add semantic_facts.pinned (A5 identity-level durability, P4a) if missing.

        A 0/1 flag marking a fact the agent (via the pin tool action) or config has declared
        identity-level / never-forget. A pinned fact is excluded from EVERY forgetting path —
        cycle decay (apply_cycle_decay), staleness decay (apply_staleness_decay), dormant-prune
        (prune_weak_facts), AND long-tier-cap eviction (enforce_long_tier_cap) — so it is the one
        kind of fact the SYSTEM will never let fade. (Pinning only PROTECTS; it never bumps
        resonance, so it can't be used to make a fact runaway-immortal — the system still owns
        decay/forget/delete for everything unpinned.) NOT NULL INTEGER default 0; legacy rows
        default 0 (nothing is pinned until explicitly asked). Self-detecting + idempotent like the
        sibling column migrations. No wall-clock involved.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "pinned" not in cols:
                    self._conn.execute(
                        "ALTER TABLE semantic_facts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
                    )
                    self._conn.commit()
                    logger.info("Migration: added semantic_facts.pinned")
            except Exception as e:
                logger.error("Migration (pinned) failed: %s", e)


    def _migrate_add_relations(self) -> None:
        """Create the Phase-5 fact_relations table (idempotent, table-only).

        Holds (subject, relation, object) triples extracted from a fact's content
        plus an optional bound-HRR encoding of the triple (holographic.encode_triple)
        for Phase-5b relational recall. Mirrors the tool_episodes migration: the
        table is created ONLY here (not in _init_db), so legacy and fresh DBs take
        the same path and _init_db stays untouched.

        fact_id is a CASCADE self-FK — pruning a fact drops its triples automatically
        (foreign_keys is ON per-connection). Subject/object are stored normalized
        (lowercased) like entity-graph keys, so the plain subject/object indexes
        serve Phase-5b lookups directly. The UNIQUE(fact_id, subject, relation,
        object) constraint makes re-extraction idempotent (INSERT OR IGNORE).
        confidence gates noisy triples downstream. No wall-clock involved.
        """
        with self._lock:
            try:
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS fact_relations (
                        relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fact_id     INTEGER REFERENCES semantic_facts(id) ON DELETE CASCADE,
                        subject     TEXT NOT NULL,
                        relation    TEXT NOT NULL,
                        object      TEXT NOT NULL,
                        confidence  REAL DEFAULT 1.0,
                        hrr_vector  BLOB,
                        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (fact_id, subject, relation, object)
                    );
                    CREATE INDEX IF NOT EXISTS idx_fact_relations_subject
                        ON fact_relations(subject);
                    CREATE INDEX IF NOT EXISTS idx_fact_relations_object
                        ON fact_relations(object);
                    CREATE INDEX IF NOT EXISTS idx_fact_relations_fact
                        ON fact_relations(fact_id);
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (fact_relations) failed or already exists: %s", e)


    def _migrate_add_agent_identity(self) -> None:
        """Create the Phase-7 agent_identity table (idempotent, table-only).

        A SEPARATE, deliberate self-model store — the agent's curated identity,
        capabilities, and standing relationship with the user. Kept OUT of
        semantic_facts ON PURPOSE: the autonomous ingest path (add_or_reinforce_fact)
        only ever touches semantic_facts, so it can NEVER reach this table. The only
        write paths are the explicit, primary-context-only set_self_model action and
        config seeding — so the self-model can't become a backdoor for the
        self-referential chatter the Phase-E gate exists to suppress. key is the PK
        (UPSERT via INSERT OR REPLACE); updated_cycle is the logical clock, never
        wall-clock. Self-detecting + idempotent like the sibling table migrations.
        """
        with self._lock:
            try:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_identity (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_cycle INTEGER
                    )
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (agent_identity) failed or already exists: %s", e)


    def _migrate_add_session_summaries(self) -> None:
        """Create the Phase-8 session_summaries table (idempotent, table-only).

        The autobiographical layer: a durable, bounded thread of 'what we did
        together' across sessions. Each row is a one-paragraph LLM gist of a session,
        generated at session end and stamped with the memory_cycle range it spans.
        A SEPARATE table (not semantic_facts / episodes) so it SURVIVES episode
        pruning — episodes are L1/ephemeral, narrative is the durable story. Framed
        as summary, never verbatim; bounded by narrative_keep. Self-detecting +
        idempotent like the sibling table migrations; cycle-stamped, never wall-clock.
        """
        with self._lock:
            try:
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        summary_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id    TEXT,
                        summary       TEXT NOT NULL,
                        started_cycle INTEGER,
                        ended_cycle   INTEGER,
                        created_cycle INTEGER,
                        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_session_summaries_cycle
                        ON session_summaries(created_cycle);
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (session_summaries) failed or already exists: %s", e)


    def _migrate_add_semantic_he(self) -> None:
        """Create the E2 blind-store semantic_he table (idempotent, table-only).

        Holds the per-fact CKKS ciphertext of the embedding for the Tier-1
        homomorphic blind store (ENCRYPTION_ROADMAP §8 E2). One opaque blob per
        fact: id is the PK *and* a CASCADE FK to semantic_facts(id), so pruning a
        fact drops its ciphertext automatically (foreign_keys is ON per-connection)
        — the blind analogue of the semantic_vec cleanup trigger. he_version stamps
        the CKKS-params version each ct was written under (lockstep with
        he_crypto.HE_PARAMS_VERSION) so a ct from old params is identifiable.

        Created unconditionally like the sibling table-only migrations — on a
        non-blind store it is simply an empty table that costs nothing; it is only
        ever populated on the blind write path (store_blind.store_he_vector), which
        the client drives when encryption_mode=blind. Created ONLY here (not in
        _init_db), so legacy and fresh DBs take the same path. No wall-clock.
        """
        with self._lock:
            try:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_he (
                        id         INTEGER PRIMARY KEY REFERENCES semantic_facts(id) ON DELETE CASCADE,
                        ct         BLOB NOT NULL,
                        he_version INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (semantic_he) failed or already exists: %s", e)


    def _migrate_add_semantic_he_hrr(self) -> None:
        """Create the E4 blind-store semantic_he_hrr table (idempotent, table-only).

        The HRR analogue of semantic_he: holds the per-fact CKKS ciphertext of the HRR
        LIFT (holographic.hrr_lift — the (cos φ, sin φ)/√dim representation whose cosine IS
        the HRR phase-similarity, ENCRYPTION_ROADMAP §8 E4). Storing the lift means the
        blind store computes HRR similarity (conflict / relational fuzzy recall) with the
        SAME homomorphic cosine it uses for embeddings — no new crypto. Same shape and
        CASCADE-FK-on-id as semantic_he, so pruning a fact drops its HRR ciphertext too.
        Created unconditionally like the sibling table-only migrations; only the blind HRR
        write path populates it.
        """
        with self._lock:
            try:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_he_hrr (
                        id         INTEGER PRIMARY KEY REFERENCES semantic_facts(id) ON DELETE CASCADE,
                        ct         BLOB NOT NULL,
                        he_version INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (semantic_he_hrr) failed or already exists: %s", e)


    def _migrate_add_semantic_he_meta(self) -> None:
        """Create the E5 5b blind-maintenance table semantic_he_meta (idempotent, table-only).

        Holds the per-fact CKKS ciphertext of the SCALED resonance scalar (resonance /
        max_resonance ∈ ~[0,1]) so the dream cycle runs blind: the store homomorphically
        DECAYS it every cycle (BlindMaintenance.decay) without ever reading it, and the
        trusted client decrypts it on a visit to settle promotion/eviction (client-assisted,
        roadmap §8 E5 5b). Same shape + CASCADE-FK-on-id as semantic_he, so pruning a fact
        drops its encrypted resonance too. Created unconditionally like the siblings; only the
        blind maintenance path populates it.
        """
        with self._lock:
            try:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_he_meta (
                        id         INTEGER PRIMARY KEY REFERENCES semantic_facts(id) ON DELETE CASCADE,
                        ct         BLOB NOT NULL,
                        he_version INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (semantic_he_meta) failed or already exists: %s", e)


    def _migrate_add_semantic_he_entities(self) -> None:
        """Create the E7 7b blind-entity table semantic_he_entities (idempotent, table-only).

        Holds the per-fact AEAD-encrypted entity-NAME set (one opaque blob per fact, random
        GCM nonce). The untrusted store can't read entity names AND identical entity sets are
        indistinguishable on disk (no deterministic token → the store learns NO entity
        co-occurrence; roadmap §7.4, client-side-no-leak posture). Overlap / conflict detection
        run CLIENT-side on the decrypted sets (retrieval.BlindEntityStore). Same shape +
        CASCADE-FK-on-id as semantic_he; only the blind entity-write path populates it.
        """
        with self._lock:
            try:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_he_entities (
                        id         INTEGER PRIMARY KEY REFERENCES semantic_facts(id) ON DELETE CASCADE,
                        ct         BLOB NOT NULL,
                        he_version INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (semantic_he_entities) failed or already exists: %s", e)


    def _migrate_add_reencrypt_audit(self) -> None:
        """Create the E6 re-encryption audit table (idempotent, table-only).

        Persists the §7.2 re-encryption audit log: one row per blind-recall grant the store
        honored — the logical memory ``cycle`` (the policy clock), the binding
        ``query_token`` from ScopeLimiter.authorize, and ``k`` results re-encrypted. The
        user-reviewable trail that makes the policy bound on the honest seam auditable rather
        than implicit (the store re-encrypts query results to the agent; this records what it
        did). ``cycle`` is logical; ``created_at`` is wall-clock display only (mirrors
        semantic_he). Created unconditionally like the sibling table-only migrations — an
        empty, free table on a non-blind store; only the blind PRE runtime path writes rows.
        """
        with self._lock:
            try:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS reencrypt_audit (
                        cycle       INTEGER NOT NULL,
                        query_token TEXT NOT NULL,
                        k           INTEGER NOT NULL,
                        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (reencrypt_audit) failed or already exists: %s", e)


    def _migrate_vec_to_cosine(self) -> None:
        """Rebuild semantic_vec with distance_metric=cosine if it is still L2.

        NON-DESTRUCTIVE: the stored float32 vectors are copied across verbatim
        (byte-for-byte) — no re-embedding, no data loss. Only the *distance
        interpretation* changes from L2 to cosine, which is what makes
        (1.0 - distance) a valid similarity. Idempotent: once the schema
        contains 'cosine' this is a no-op, so it is safe on every open.

        REVERSIBILITY: this changes the table's metric in place. The vectors
        themselves are unchanged, but to revert the metric you would restore
        from the pre-migration backup (recommended before first run). There is
        no in-DB "undo" because vec0 bakes the metric into the table definition.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_vec'"
                ).fetchone()
                sql = (row["sql"] if row else "") or ""
                if not sql or "cosine" in sql.lower():
                    return  # nothing to migrate / already cosine

                m = re.search(r"float\[(\d+)\]", sql)
                stored_dim = int(m.group(1)) if m else self.vector_dim

                logger.warning(
                    "Migration: rebuilding semantic_vec with distance_metric=cosine "
                    "(was L2). Copying existing %d-d vectors verbatim — no re-embed.",
                    stored_dim,
                )

                vectors = [
                    (r["id"], r["embedding"])
                    for r in self._conn.execute("SELECT id, embedding FROM semantic_vec").fetchall()
                ]
                # SAFEGUARD: snapshot the whole DB before the destructive
                # DROP + rebuild. VACUUM INTO is WAL-aware and atomic, so an
                # interrupted rebuild leaves a restorable .pre-cosine.bak copy.
                try:
                    import os as _os
                    import time as _time
                    backup_path = f"{self.db_path}.pre-cosine.bak"
                    if _os.path.exists(backup_path):
                        # A stale snapshot from an interrupted prior attempt would
                        # make VACUUM INTO fail forever ("output file already
                        # exists"). Preserve it under a suffixed name — it may be
                        # the only intact pre-crash copy — and snapshot fresh.
                        aside = f"{backup_path}.{int(_time.time())}"
                        _os.replace(backup_path, aside)
                        logger.warning(
                            "Migration: stale pre-cosine backup moved aside to %s",
                            aside,
                        )
                    self._conn.execute("VACUUM INTO ?", (backup_path,))
                    logger.warning("Migration: pre-cosine DB snapshot written to %s", backup_path)
                except Exception as _be:
                    logger.error(
                        "Migration: pre-cosine backup FAILED (%s) — skipping cosine "
                        "rebuild to avoid unprotected data loss.", _be
                    )
                    return
                self._conn.execute("DROP TABLE semantic_vec")
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE semantic_vec USING vec0("
                    f"id INTEGER PRIMARY KEY, "
                    f"embedding float[{stored_dim}] distance_metric=cosine)"
                )
                for fid, emb in vectors:
                    if emb is None:
                        continue
                    self._conn.execute(
                        "INSERT INTO semantic_vec (id, embedding) VALUES (?, ?)",
                        (fid, emb),
                    )
                self._conn.commit()
                copied = self._conn.execute(
                    "SELECT COUNT(*) FROM semantic_vec"
                ).fetchone()[0]
                expected = sum(1 for _, e in vectors if e is not None)
                if copied != expected:
                    # Half-built index — do NOT trust it. Fall back to FTS-only for
                    # this session and shout loudly; the verified snapshot at
                    # {db}.pre-cosine.bak can be restored manually.
                    self.degraded = True
                    logger.critical(
                        "Migration: cosine rebuild copied %d of %d vectors — MISMATCH. "
                        "Entering DEGRADED (FTS-only) mode for safety. Restore from "
                        "%s.pre-cosine.bak before next run.",
                        copied, expected, self.db_path,
                    )
            except Exception as e:
                logger.error("Migration (cosine rebuild) failed: %s", e)

                
    def _migrate_fts_trigger(self) -> None:
        """Recreate semantic_facts_au with a WHEN guard on existing DBs.

        CREATE TRIGGER IF NOT EXISTS cannot replace an already-present trigger,
        so drop and recreate it gated. Idempotent and cheap.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type='trigger' AND name='semantic_facts_au'"
                ).fetchone()
                sql = (row["sql"] if row else "") or ""
                if "WHEN" in sql.upper():
                    return  # already gated
                self._conn.execute("DROP TRIGGER IF EXISTS semantic_facts_au")
                self._conn.execute(
                    """
                    CREATE TRIGGER semantic_facts_au AFTER UPDATE ON semantic_facts
                    WHEN old.content IS NOT new.content OR old.category IS NOT new.category
                    BEGIN
                        INSERT INTO semantic_fts(semantic_fts, rowid, content, category)
                            VALUES ('delete', old.id, old.content, old.category);
                        INSERT INTO semantic_fts(rowid, content, category)
                            VALUES (new.id, new.content, new.category);
                    END
                    """
                )
                self._conn.commit()
                logger.info("Migration: gated semantic_facts_au FTS trigger with WHEN clause")
            except Exception as e:
                logger.error("Migration (FTS trigger gate) failed: %s", e)                


    def _migrate_delete_trigger(self) -> None:
        """Ensure semantic_facts_ad cleans semantic_vec + fact_entities on delete.

        CREATE TRIGGER IF NOT EXISTS can't replace an older, partial trigger, so
        drop and recreate the full version. Idempotent: skips when both DELETE
        statements are already present.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type='trigger' AND name='semantic_facts_ad'"
                ).fetchone()
                sql = (row["sql"] if row else "") or ""
                up = sql.upper()
                if "SEMANTIC_VEC" in up and "FACT_ENTITIES" in up:
                    return  # already the full cleanup version
                self._conn.execute("DROP TRIGGER IF EXISTS semantic_facts_ad")
                self._conn.execute(
                    """
                    CREATE TRIGGER semantic_facts_ad AFTER DELETE ON semantic_facts BEGIN
                        INSERT INTO semantic_fts(semantic_fts, rowid, content, category)
                            VALUES ('delete', old.id, old.content, old.category);
                        DELETE FROM semantic_vec WHERE id = old.id;
                        DELETE FROM fact_entities WHERE fact_id = old.id;
                    END
                    """
                )
                self._conn.commit()
                logger.info("Migration: rebuilt semantic_facts_ad with vec + entity cleanup")
            except Exception as e:
                logger.error("Migration (delete trigger) failed: %s", e)


    def _migrate_add_abstraction_sources(self) -> None:
        """Create/upgrade abstraction_sources with a surrogate PK (idempotent).

        The original composite PK (abstract_id, source_id) is incompatible with
        ON DELETE SET NULL once FK enforcement is on: a second source nulling out
        produces a duplicate (abstract_id, NULL). Use a surrogate rowid PK and a
        UNIQUE constraint that tolerates NULLs (SQLite treats NULLs as distinct
        in UNIQUE), then copy any existing rows across.
        """
        with self._lock:
            try:
                cols = {
                    r["name"]
                    for r in self._conn.execute(
                        "PRAGMA table_info(abstraction_sources)"
                    ).fetchall()
                }
                # Fresh DB: create the correct shape directly.
                if not cols:
                    self._conn.execute("""
                        CREATE TABLE abstraction_sources (
                            link_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                            abstract_id INTEGER REFERENCES semantic_facts(id) ON DELETE CASCADE,
                            source_id   INTEGER REFERENCES semantic_facts(id) ON DELETE SET NULL,
                            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            cluster_size_at_creation INTEGER,
                            UNIQUE (abstract_id, source_id)
                        )
                    """)
                    self._conn.commit()
                    return
                # Existing DB with the old composite-PK shape: migrate in place.
                if "link_id" not in cols:
                    self._conn.execute("ALTER TABLE abstraction_sources RENAME TO abstraction_sources_old")
                    self._conn.execute("""
                        CREATE TABLE abstraction_sources (
                            link_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                            abstract_id INTEGER REFERENCES semantic_facts(id) ON DELETE CASCADE,
                            source_id   INTEGER REFERENCES semantic_facts(id) ON DELETE SET NULL,
                            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            cluster_size_at_creation INTEGER,
                            UNIQUE (abstract_id, source_id)
                        )
                    """)
                    self._conn.execute("""
                        INSERT INTO abstraction_sources
                            (abstract_id, source_id, created_at, cluster_size_at_creation)
                        SELECT abstract_id, source_id, created_at, cluster_size_at_creation
                        FROM abstraction_sources_old
                    """)
                    self._conn.execute("DROP TABLE abstraction_sources_old")
                    self._conn.commit()
                    logger.info("Migration: rebuilt abstraction_sources with surrogate PK")
            except Exception as e:
                logger.debug("Migration (abstraction_sources) failed or already exists: %s", e)


    def _migrate_add_tool_episodes(self) -> None:
        """Create/upgrade the tool_episodes procedural log (idempotent).

        Raw tool-invocation events live here (the procedural L1 layer). They are
        NOT semantic facts: no dedup, no HRR, no per-event resonance. The dream
        cycle later generalizes them into 'procedural' facts in semantic_facts.

        call_id carries the platform tool_call_id and is enforced UNIQUE via a
        partial index so restart-driven replays of the full message history are
        rejected at the DB layer (INSERT OR IGNORE in add_tool_episode). The
        partial index (WHERE call_id IS NOT NULL) keeps legacy rows — which
        predate the column and hold NULL — out of the uniqueness check.
        """
        with self._lock:
            try:
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS tool_episodes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        tool_name TEXT NOT NULL,
                        arguments TEXT,
                        result TEXT,
                        success INTEGER DEFAULT 0,
                        memory_cycle INTEGER DEFAULT 0,
                        distilled INTEGER DEFAULT 0,
                        call_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_tool_episodes_tool
                        ON tool_episodes(tool_name);
                    CREATE INDEX IF NOT EXISTS idx_tool_episodes_undistilled
                        ON tool_episodes(distilled) WHERE distilled = 0;
                """)
                # Upgrade pre-existing tables that lack call_id.
                cols = {
                    r["name"]
                    for r in self._conn.execute(
                        "PRAGMA table_info(tool_episodes)"
                    ).fetchall()
                }
                if cols and "call_id" not in cols:
                    self._conn.execute(
                        "ALTER TABLE tool_episodes ADD COLUMN call_id TEXT"
                    )
                    logger.info("Migration: added tool_episodes.call_id")
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_episodes_call_id "
                    "ON tool_episodes(call_id) WHERE call_id IS NOT NULL"
                )
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (tool_episodes) failed or already exists: %s", e)


    def _migrate_add_canonical_facts(self) -> None:
        """Create the optional canonical-state projection table (idempotent, table-only).

        A SEPARATE current-value layer 'over' the lattice: each row is key -> current
        value with provenance (source_fact_id), temporal validity (valid_from/until_cycle),
        a supersession chain, and a review_status. Lets an agent ask 'what is the current
        value of X' as one canonical field instead of inferring it from recall ranking.
        Created unconditionally like the sibling table-only migrations — empty and free on
        a store that never uses it; only the explicit set_canonical path writes rows.
        Cycle-stamped, never wall-clock. The partial index serves the hot 'current row for
        key' lookup (valid_until_cycle IS NULL)."""
        with self._lock:
            try:
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS canonical_facts (
                        canonical_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                        key               TEXT NOT NULL,
                        value             TEXT NOT NULL,
                        category          TEXT DEFAULT 'general',
                        source_fact_id    INTEGER REFERENCES semantic_facts(id) ON DELETE SET NULL,
                        valid_from_cycle  INTEGER,
                        valid_until_cycle INTEGER,
                        superseded_by     INTEGER REFERENCES canonical_facts(canonical_id) ON DELETE SET NULL,
                        review_status     TEXT DEFAULT 'unreviewed',
                        created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_canonical_current
                        ON canonical_facts(key) WHERE valid_until_cycle IS NULL;
                    CREATE INDEX IF NOT EXISTS idx_canonical_key
                        ON canonical_facts(key);
                """)
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (canonical_facts) failed or already exists: %s", e)


    def _migrate_add_write_batches(self) -> None:
        """Create the write-batch provenance table + semantic_facts.batch_id (idempotent).

        Semantic ROLLBACK: a consolidation epoch or dream cycle opens a 'batch' and every
        fact it WRITES is stamped with batch_id, so a bad generative run (weak extraction
        model, malformed transcript, model regression) can be reviewed and rolled back as a
        unit instead of by manual row cleanup. write_batches records phase / model / session /
        config_hash + the logical cycle and a status; rollback flips the status and deletes the
        batch's non-pinned facts (pinned = a deliberate user lock, kept). batch_id is a plain
        nullable INTEGER (NULL = a normal user/agent write, not part of a batch). Created
        unconditionally like the sibling table-only migrations; cycle-stamped, never wall-clock."""
        with self._lock:
            try:
                self._conn.executescript("""
                    CREATE TABLE IF NOT EXISTS write_batches (
                        batch_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        phase          TEXT NOT NULL,
                        source_session TEXT,
                        model          TEXT,
                        config_hash    TEXT,
                        created_cycle  INTEGER,
                        n_writes       INTEGER NOT NULL DEFAULT 0,
                        status         TEXT NOT NULL DEFAULT 'active',
                        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        closed_at      TIMESTAMP,
                        rolled_back_at TIMESTAMP
                    );
                """)
                cols = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(semantic_facts)").fetchall()
                }
                if "batch_id" not in cols:
                    self._conn.execute("ALTER TABLE semantic_facts ADD COLUMN batch_id INTEGER")
                    logger.info("Migration: added semantic_facts.batch_id")
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_semantic_facts_batch "
                    "ON semantic_facts(batch_id) WHERE batch_id IS NOT NULL")
                self._conn.commit()
            except Exception as e:
                logger.debug("Migration (write_batches) failed or already exists: %s", e)


    def _stamp_meta(self) -> None:
        """Record encoding metadata; warn (don't crash) on hrr_dim drift."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES ('hrr_dim', ?)",
                    (str(self.hrr_dim),),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('encoding_version', 'rich-v2')"
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('vec_metric', 'cosine')"
                )
                self._conn.commit()

                row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='hrr_dim'"
                ).fetchone()
                if row and int(row["value"]) != self.hrr_dim:
                    logger.error(
                        "HRR DIM MISMATCH: DB stamped hrr_dim=%s but configured hrr_dim=%d. "
                        "Existing HRR vectors will be SKIPPED (length-guarded) until a re-encode "
                        "migration runs. Set hrr_dim=%s in config to use the stored vectors.",
                        row["value"], self.hrr_dim, row["value"],
                    )
            except Exception as e:
                logger.debug("Meta stamping failed (non-fatal): %s", e)
