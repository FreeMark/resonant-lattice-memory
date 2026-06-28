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
from typing import List, Optional, Tuple

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
# per-fact entity-name set (E7 7b — opaque blob, overlap is a client-side op).
_HE_TABLES = ("semantic_he", "semantic_he_hrr", "semantic_he_meta", "semantic_he_entities")


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
}


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
