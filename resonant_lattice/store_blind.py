"""store_blind.py — BlindMixin: Tier-1 blind-store (HE) ciphertext storage.

The STORE side of the homomorphic blind store (ENCRYPTION_ROADMAP §8). Holds ONLY opaque
ciphertext per fact across the four blind tables — `semantic_he` (CKKS embedding, E2),
`semantic_he_hrr` (CKKS HRR lift, E4), `semantic_he_meta` (CKKS resonance scalar, E5), and
`semantic_he_entities` (AEAD entity-name set, E7) — plus the `reencrypt_audit` log. The store
never decrypts (it has no secret key). Encryption happens client-side — e.g.
`he_crypto.BlindRecallPRE.encrypt_unit_vector` under the public key, or AEAD for entities —
BEFORE the ct reaches here, so these methods are pure SQLite BLOB ops with no `openfhe`
dependency on the store side, and are fully substrate-testable without HE installed.

Gating: the blind tables are created unconditionally by the schema migrations (empty tables
cost nothing on non-blind stores, mirroring the other table-only migrations), but they are only
ever populated on the blind write path, which the client/provider drives when
`encryption_mode=blind`. These methods are mode-agnostic — the caller decides whether to use
them; the `table` selector is allowlisted (`_he_table`) so an untrusted name is never
interpolated into SQL.

Mixed into LatticeStore; uses self._conn/_lock like the sibling store_* mixins and
never imports the composite (flat sibling imports only)."""

import logging
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bump in lockstep with he_crypto.HE_PARAMS_VERSION when the CKKS params change, so a
# ciphertext written under old params is identifiable (and rejectable) on read.
DEFAULT_HE_VERSION = 1

# Blind-vector ciphertext tables. semantic_he holds the encrypted EMBEDDING (E2);
# semantic_he_hrr holds the encrypted HRR LIFT (E4). Same shape + ops, so the methods
# below take a `table` selector — allowlisted (NEVER interpolate an untrusted name into SQL).
DEFAULT_HE_TABLE = "semantic_he"
# semantic_he = encrypted embedding (E2); semantic_he_hrr = encrypted HRR lift (E4);
# semantic_he_meta = encrypted resonance scalar (E5 5b); semantic_he_entities = AEAD-encrypted
# per-fact entity-name set (E7 7b — opaque blob, overlap is a client-side op);
# semantic_he_content = AEAD content surface ({content, category, quote, source}, §5-1); and the
# §5-1b sealed TEXT surfaces keyed by their SOURCE row (not fact id): semantic_he_episodes
# ({role, content} from episodes), semantic_he_triples ({subject, relation, object} from
# fact_relations), semantic_he_summaries (summary text from session_summaries).
_HE_TABLES = ("semantic_he", "semantic_he_hrr", "semantic_he_meta", "semantic_he_entities",
              "semantic_he_content", "semantic_he_episodes", "semantic_he_triples",
              "semantic_he_summaries")


def _he_table(table: str) -> str:
    if table not in _HE_TABLES:
        raise ValueError(f"unknown HE table {table!r} (expected one of {_HE_TABLES})")
    return table


# Per-table "plaintext SOURCE present" predicate for the reconciliation worklist. A fact only
# belongs on a table's missing-blind worklist if the plaintext it would be mirrored FROM actually
# exists — otherwise it can NEVER be mirrored and would permanently saturate the capped LIMIT
# window (the write-path-completeness poison-pill: a fact with a NULL hrr_vector is returned every
# pass, skipped every pass, and starves higher-id facts that DO have an HRR lift). These fragments
# are STATIC and code-controlled — keyed by the already-allowlisted table name, never an untrusted
# string interpolated into SQL.
_HE_SOURCE_PRESENT = {
    "semantic_he": "EXISTS (SELECT 1 FROM semantic_vec v WHERE v.id = f.id)",
    "semantic_he_hrr": "f.hrr_vector IS NOT NULL",
    "semantic_he_meta": None,        # resonance is always settable
    "semantic_he_entities": None,    # an empty set is mirrorable (set_entities([]) writes a row)
    "semantic_he_content": None,     # every fact has content -> always mirrorable (§5-1)
}


# ── streaming blind-recall scan sizing (A1) ──────────────────────────────────────────
# The blind recall scan scores the query against EVERY stored ciphertext. iter_he_vectors
# fetchall's the whole table, so at ~1 MB per embedding ciphertext a real corpus lands
# GIGABYTES resident and the scan thrashes the memory bus (measured: 4,952 facts => ~5 GB
# RSS, and adding CPU threads made it SLOWER — bandwidth-bound). stream_he_vectors pages
# instead, holding only `batch` ciphertexts at once. Latency is flat across batch size, so
# the batch is purely a RAM/concurrency dial; resolve_scan_batch sizes it PORTABLY from the
# MEASURED per-ciphertext footprint and DETECTED available RAM — nothing hardcoded to a host.
# Latency is FLAT across batch size (measured: B=64/256/1024 all ~275s), so a bigger batch buys
# nothing but RAM. The tuner therefore DEFAULTS to a modest latency-optimal page and only SHRINKS
# it when RAM (per concurrent scan) can't afford even that — it never inflates to fill memory,
# which on a big-RAM host would just recreate the fetchall problem (whole corpus resident) and
# starve concurrency. Smaller batch => more scans fit in RAM at once.
_SCAN_BATCH_TARGET = 256       # round-trip-amortized sweet spot; the default page size
_SCAN_BATCH_MIN = 32           # never page fewer than this (keep DB round-trips amortized)
_SCAN_RAM_FRACTION = 0.20      # of MemAvailable, per concurrent scan — a CEILING, not a target


def detect_available_ram_bytes() -> Optional[int]:
    """Best-effort available RAM in bytes (Linux ``/proc/meminfo`` MemAvailable). Returns
    None on hosts without it (e.g. the non-Linux dev box) so callers fall back to the safe
    default batch. Blind recall only runs where openfhe is installed (Linux), so the
    common path resolves; this stays dependency-free and never raises."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def resolve_scan_batch(configured: int, per_ct_bytes: int, count: int,
                       avail_ram_bytes: Optional[int], concurrency: int = 1) -> int:
    """Choose the streaming-scan batch size.

    A positive ``configured`` is an explicit override and wins. Otherwise AUTO: start at the
    latency-optimal ``_SCAN_BATCH_TARGET`` and SHRINK it (down to ``_SCAN_BATCH_MIN``) only if a
    fraction of DETECTED available RAM, split across the intended ``concurrency``, can't hold
    that many MEASURED-size ciphertexts. It never GROWS the batch to fill RAM: latency is flat
    across batch size, so a bigger page would only waste memory and starve concurrent scans (and
    on a big-RAM host recreate the whole-corpus-resident fetchall problem). Never larger than the
    corpus. Portable: adapts DOWN on small/busy hosts, nothing hardcoded to a machine."""
    if configured and int(configured) > 0:
        batch = int(configured)
    else:
        batch = _SCAN_BATCH_TARGET
        if per_ct_bytes and avail_ram_bytes:
            ram_budget = avail_ram_bytes * _SCAN_RAM_FRACTION / max(1, int(concurrency))
            batch = min(batch, int(ram_budget / per_ct_bytes))
        batch = max(_SCAN_BATCH_MIN, batch)
    if count and int(count) > 0:
        batch = min(batch, int(count))
    return max(1, batch)


class BlindMixin:

    def store_he_vector(self, fact_id: int, ct_blob: bytes,
                        he_version: int = DEFAULT_HE_VERSION,
                        table: str = DEFAULT_HE_TABLE) -> None:
        """Persist the CKKS ciphertext of a fact's blind vector (the Tier-1 blind write).

        ``table`` selects ``semantic_he`` (embedding, E2) or ``semantic_he_hrr`` (HRR lift,
        E4). A pure opaque-blob insert: the store holds no key and never inspects the
        plaintext. INSERT OR REPLACE keyed on the fact id, so re-embedding a fact overwrites
        its single ct. Raises ValueError on an empty/non-bytes blob so a silently-dropped
        vector can't masquerade as a stored one.
        """
        if not isinstance(ct_blob, (bytes, bytearray)) or not ct_blob:
            raise ValueError("ct_blob must be non-empty bytes")
        tbl = _he_table(table)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {tbl} (id, ct, he_version) VALUES (?, ?, ?)",
                (int(fact_id), bytes(ct_blob), int(he_version)),
            )
            self._conn.commit()

    def get_he_vector(self, fact_id: int, table: str = DEFAULT_HE_TABLE) -> Optional[bytes]:
        """Return the stored ciphertext blob for one fact, or None if absent."""
        tbl = _he_table(table)
        with self._lock:
            row = self._conn.execute(
                f"SELECT ct FROM {tbl} WHERE id = ?", (int(fact_id),)
            ).fetchone()
        return bytes(row["ct"]) if row else None

    def iter_he_vectors(self, table: str = DEFAULT_HE_TABLE) -> List[Tuple[int, bytes]]:
        """Return [(fact_id, ct_blob), …] for every stored ct — the blind-recall scan.

        Materialized list (not a live cursor) so the shared connection isn't held
        open across the caller's homomorphic scoring loop. Ordered by id for stable,
        reproducible scans.
        """
        tbl = _he_table(table)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, ct FROM {tbl} ORDER BY id"
            ).fetchall()
        return [(r["id"], bytes(r["ct"])) for r in rows]

    def stream_he_vectors(self, table: str = DEFAULT_HE_TABLE,
                          batch: int = _SCAN_BATCH_TARGET) -> Iterator[Tuple[int, bytes]]:
        """Yield ``(fact_id, ct_blob)`` for every stored ct in id order, ``batch`` rows at a
        time — the BOUNDED-RAM blind-recall scan (A1). Unlike iter_he_vectors (which
        fetchall's the whole table, so ~1 MB/ct becomes GBs resident at corpus scale), this
        keeps only ``batch`` ciphertexts in memory at once via KEYED PAGINATION
        (``WHERE id > last ORDER BY id LIMIT batch`` over the primary key — O(log n) per page,
        no OFFSET rescan). It reuses ``self._conn`` (so it composes with at-rest SQLCipher; a
        fresh connection would lack the unlocked key) and holds ``self._lock`` only for each
        brief page fetch, releasing it while the caller does the slow homomorphic scoring."""
        tbl = _he_table(table)
        batch = max(1, int(batch))
        last_id: Optional[int] = None
        while True:
            with self._lock:
                if last_id is None:
                    rows = self._conn.execute(
                        f"SELECT id, ct FROM {tbl} ORDER BY id LIMIT ?", (batch,)
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id, ct FROM {tbl} WHERE id > ? ORDER BY id LIMIT ?",
                        (last_id, batch),
                    ).fetchall()
            if not rows:
                break
            for r in rows:
                yield (int(r["id"]), bytes(r["ct"]))
            last_id = int(rows[-1]["id"])

    def he_blob_size(self, table: str = DEFAULT_HE_TABLE) -> int:
        """Average stored ciphertext size in bytes (0 if the table is empty) — the MEASURED
        per-ct footprint that ``resolve_scan_batch`` uses to size the streaming scan. Cheap
        AVG(LENGTH) over the (small) blind table."""
        tbl = _he_table(table)
        with self._lock:
            row = self._conn.execute(f"SELECT AVG(LENGTH(ct)) FROM {tbl}").fetchone()
        return int(row[0]) if row and row[0] else 0

    def count_he_vectors(self, table: str = DEFAULT_HE_TABLE) -> int:
        """Number of stored ciphertexts — leaks only the fact count (see §7.3)."""
        tbl = _he_table(table)
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

    def facts_missing_blind(self, table: str = DEFAULT_HE_TABLE, limit: int = 0) -> List[int]:
        """Fact ids with a (non-superseded) plaintext fact row but NO ciphertext in ``table`` —
        the blind-tier RECONCILIATION worklist (roadmap §14 Priority 6a / write-path completeness).

        A LEFT JOIN so every fact created OUTSIDE the consolidation mirror hook — abstraction /
        gist / procedural distillation (all store-side) + the builtin-memory mirror — is caught,
        plus a first-blind-enable BACKFILL of a pre-existing store. The provider's
        ``_blind_reconcile`` reads each id's embedding/HRR/entities back from the plaintext store
        and mirrors them, so this is the idempotent driver (once mirrored, an id drops off the
        list). ``table`` is allowlisted via ``_he_table`` (never interpolate an untrusted name);
        ``limit`` > 0 batches a large backfill so a single cycle never mirrors an unbounded set.
        Ordered by id for stable, resumable batching."""
        tbl = _he_table(table)
        where = "b.id IS NULL AND f.tier != 'superseded'"
        src = _HE_SOURCE_PRESENT.get(table)
        if src:
            where += f" AND {src}"
        sql = (f"SELECT f.id FROM semantic_facts f "
               f"LEFT JOIN {tbl} b ON b.id = f.id "
               f"WHERE {where} ORDER BY f.id")
        if int(limit) > 0:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [r["id"] for r in self._conn.execute(sql).fetchall()]

    def facts_needing_entity_mirror(self, limit: int = 0) -> List[int]:
        """Fact ids whose AEAD entity set needs (re)mirroring into ``semantic_he_entities``:
        either NO ciphertext row yet, OR ``entities_dirty = 1`` — a new entity link was added
        since the last mirror (reinforcement grows a fact's entity set; see
        ``store_facts._link_entities``).

        Distinct from ``facts_missing_blind`` because the entity set is the ONE blind source that
        MUTATES — the embedding and HRR lift are content-derived and immutable, so 'mirror once
        when the row is missing' silently goes stale for entities. ``_blind_reconcile`` mirrors
        each id and clears the flag via ``mark_entities_mirrored``, keeping this the idempotent
        driver. ``limit`` > 0 batches a backfill; ordered by id for stable, resumable batching."""
        with self._lock:
            sql = ("SELECT f.id FROM semantic_facts f "
                   "LEFT JOIN semantic_he_entities b ON b.id = f.id "
                   "WHERE f.tier != 'superseded' AND (b.id IS NULL OR f.entities_dirty = 1) "
                   "ORDER BY f.id")
            if int(limit) > 0:
                sql += f" LIMIT {int(limit)}"
            return [r["id"] for r in self._conn.execute(sql).fetchall()]

    def mark_entities_mirrored(self, fact_id: int) -> None:
        """Clear ``entities_dirty`` after a fact's AEAD entity set has been (re)mirrored, so it
        drops off ``facts_needing_entity_mirror`` until its entity set changes again."""
        with self._lock:
            self._conn.execute(
                "UPDATE semantic_facts SET entities_dirty = 0 WHERE id = ?", (int(fact_id),)
            )
            self._conn.commit()

    # ── §5-1 sealed-content dedup identity (content_hmac, 3e) ─────────────────────
    def set_content_hmac(self, fact_id: int, hmac_hex: str) -> None:
        """Store a fact's blind dedup identity — the keyed HMAC hex of its normalized content
        (crypto_keys.content_hmac). Pure column write; the store never holds the HMAC key, so it
        can match equal identities but not derive them. Idempotent per fact (plain UPDATE)."""
        if not hmac_hex:
            raise ValueError("hmac_hex must be non-empty")
        with self._lock:
            self._conn.execute(
                "UPDATE semantic_facts SET content_hmac = ? WHERE id = ?",
                (str(hmac_hex), int(fact_id)),
            )
            self._conn.commit()

    def facts_missing_content_hmac(self, limit: int = 0) -> List[int]:
        """Fact ids (non-superseded) whose ``content_hmac`` is not yet computed — the §5-1
        dedup-identity backfill worklist. Independent of the content ciphertext mirror (a fact can
        have one without the other), so ``_blind_reconcile`` drives it as its own idempotent pass:
        once ``set_content_hmac`` stamps a row it drops off. ``limit`` > 0 batches; ordered by id
        for stable, resumable batching."""
        sql = ("SELECT id FROM semantic_facts "
               "WHERE content_hmac IS NULL AND tier != 'superseded' ORDER BY id")
        if int(limit) > 0:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [r["id"] for r in self._conn.execute(sql).fetchall()]

    # ── §5-1b sealed TEXT surfaces (episodes / triples / summaries) ───────────────
    # Each mirrors a DIFFERENT source table (not fact-keyed), so each has its own LEFT-JOIN
    # worklist keyed on the SOURCE row PK + a payload reader. The SQL fragments are static and
    # code-controlled (no untrusted interpolation). The mirror table's CASCADE-FK to the source PK
    # means pruning the source (episode prune, fact prune -> triple CASCADE, summary prune) drops
    # the ciphertext automatically — no stale blind rows.
    def episodes_missing_blind(self, limit: int = 0) -> List[int]:
        """Episode ids with a plaintext row but NO ciphertext in ``semantic_he_episodes`` — the
        §5-1b episode-mirror worklist. Idempotent (LEFT JOIN); ``limit`` > 0 batches; id order."""
        sql = ("SELECT e.id FROM episodes e LEFT JOIN semantic_he_episodes b ON b.id = e.id "
               "WHERE b.id IS NULL ORDER BY e.id")
        if int(limit) > 0:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [r["id"] for r in self._conn.execute(sql).fetchall()]

    def get_episode_payload(self, episode_id: int) -> Optional[dict]:
        """The sealable text surface of one episode — ``{role, content}`` — or None if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT role, content FROM episodes WHERE id = ?", (int(episode_id),)).fetchone()
        return {"role": row["role"], "content": row["content"]} if row else None

    def triples_missing_blind(self, limit: int = 0) -> List[int]:
        """Relation ids (fact_relations.relation_id) with a plaintext row but NO ciphertext in
        ``semantic_he_triples`` — the §5-1b triple-mirror worklist. Idempotent; ``limit`` batches."""
        sql = ("SELECT r.relation_id FROM fact_relations r "
               "LEFT JOIN semantic_he_triples b ON b.id = r.relation_id "
               "WHERE b.id IS NULL ORDER BY r.relation_id")
        if int(limit) > 0:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [r["relation_id"] for r in self._conn.execute(sql).fetchall()]

    def get_triple_payload(self, relation_id: int) -> Optional[dict]:
        """The sealable text of one triple — ``{subject, relation, object}`` — or None if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT subject, relation, object FROM fact_relations WHERE relation_id = ?",
                (int(relation_id),)).fetchone()
        return {"subject": row["subject"], "relation": row["relation"],
                "object": row["object"]} if row else None

    def summaries_missing_blind(self, limit: int = 0) -> List[int]:
        """Summary ids (session_summaries.summary_id) with a plaintext row but NO ciphertext in
        ``semantic_he_summaries`` — the §5-1b summary-mirror worklist. Idempotent; ``limit`` batches."""
        sql = ("SELECT s.summary_id FROM session_summaries s "
               "LEFT JOIN semantic_he_summaries b ON b.id = s.summary_id "
               "WHERE b.id IS NULL ORDER BY s.summary_id")
        if int(limit) > 0:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            return [r["summary_id"] for r in self._conn.execute(sql).fetchall()]

    def get_summary_payload(self, summary_id: int) -> Optional[str]:
        """The sealable text of one session summary (a string), or None if absent."""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM session_summaries WHERE summary_id = ?",
                (int(summary_id),)).fetchone()
        return row["summary"] if row else None

    # ── E6 re-encryption audit (the persisted §7.2 trail) ─────────────────────────
    def record_reencrypt_event(self, cycle: int, query_token: str, k: int) -> None:
        """Append one re-encryption grant to the persisted audit log (roadmap 6c).

        The store records WHAT it re-encrypted for the agent — the logical ``cycle``, the
        binding ``query_token`` (from ScopeLimiter.authorize / BlindReEncryptGate.register),
        and ``k`` results — so the user has a substrate-checkable trail of the honest-seam
        policy bound. Pure SQLite; no key/crypto here."""
        if not query_token or int(k) <= 0:
            raise ValueError("query_token must be non-empty and k positive")
        with self._lock:
            self._conn.execute(
                "INSERT INTO reencrypt_audit (cycle, query_token, k) VALUES (?, ?, ?)",
                (int(cycle), str(query_token), int(k)),
            )
            self._conn.commit()

    def get_reencrypt_events(self, limit: int = 100):
        """Recent re-encryption audit rows (most recent first) for review / memory_audit."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT cycle, query_token, k, created_at FROM reencrypt_audit "
                "ORDER BY rowid DESC LIMIT ?", (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_reencrypt_events(self) -> int:
        """Total re-encryption events recorded (0 on a non-blind store)."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM reencrypt_audit").fetchone()[0]
