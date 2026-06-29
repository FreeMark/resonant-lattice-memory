"""store_canonical.py — CanonicalMixin: the optional current-value projection.

Mixed into LatticeStore. A SEPARATE, explicit layer 'over' the resonance lattice:
each row is `key -> current value` with provenance + temporal validity, so an agent
can ask "what is the current value of X" as a single canonical field instead of
inferring it from recall ranking, markers, and conflict metadata.

It does NOT replace the lattice and is NOT written by any autonomous path — only the
explicit set_canonical / review_canonical calls touch it (mirrors the agent_identity
self-model store). History is preserved: updating a key closes the old row
(valid_until_cycle + superseded_by) and inserts a new current row. Cycle-stamped,
never wall-clock. Relies on the composite for self._conn, self._lock,
self.get_cycle_counts."""

import logging

logger = logging.getLogger(__name__)


class CanonicalMixin:

    def set_canonical(self, key, value, *, category: str = "general",
                      source_fact_id=None, cycle=None,
                      review_status: str = "unreviewed"):
        """Upsert the CURRENT canonical value for `key`.

        If a different current value exists it is closed (valid_until_cycle set,
        superseded_by → the new row) and the new value becomes current — history
        is preserved, not overwritten. If the value is unchanged this is a no-op
        refresh of metadata. Returns the canonical_id of the current row.
        """
        key = (key or "").strip()
        if not key or value is None:
            raise ValueError("set_canonical requires a non-empty key and a value")
        value = str(value)
        if cycle is None:
            cycle = self.get_cycle_counts()[0]
        with self._lock:
            cur = self._conn.execute(
                "SELECT canonical_id, value FROM canonical_facts "
                "WHERE key=? AND valid_until_cycle IS NULL",
                (key,),
            ).fetchone()
            if cur is not None and cur["value"] == value:
                # same current value — refresh metadata only (no new history row)
                self._conn.execute(
                    "UPDATE canonical_facts SET category=?, "
                    "source_fact_id=COALESCE(?, source_fact_id), review_status=?, "
                    "updated_at=CURRENT_TIMESTAMP WHERE canonical_id=?",
                    (category, source_fact_id, review_status, cur["canonical_id"]),
                )
                self._conn.commit()
                return cur["canonical_id"]
            new_id = self._conn.execute(
                "INSERT INTO canonical_facts(key, value, category, source_fact_id, "
                "valid_from_cycle, review_status) VALUES (?, ?, ?, ?, ?, ?)",
                (key, value, category, source_fact_id, cycle, review_status),
            ).lastrowid
            if cur is not None:
                self._conn.execute(
                    "UPDATE canonical_facts SET valid_until_cycle=?, superseded_by=?, "
                    "updated_at=CURRENT_TIMESTAMP WHERE canonical_id=?",
                    (cycle, new_id, cur["canonical_id"]),
                )
            self._conn.commit()
            return new_id

    def get_canonical(self, key):
        """The current canonical record for `key` (dict) or None. Read-only."""
        key = (key or "").strip()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT canonical_id, key, value, category, source_fact_id, "
                "valid_from_cycle, valid_until_cycle, superseded_by, review_status, "
                "created_at, updated_at FROM canonical_facts "
                "WHERE key=? AND valid_until_cycle IS NULL",
                (key,),
            ).fetchone()
            return dict(row) if row else None

    def list_canonical(self, *, category=None):
        """All current canonical records (optionally filtered by category)."""
        with self._lock:
            if category:
                rows = self._conn.execute(
                    "SELECT * FROM canonical_facts WHERE valid_until_cycle IS NULL "
                    "AND category=? ORDER BY key",
                    (category,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM canonical_facts WHERE valid_until_cycle IS NULL "
                    "ORDER BY key"
                ).fetchall()
            return [dict(r) for r in rows]

    def canonical_history(self, key):
        """Every value `key` has held, oldest first (current row last)."""
        key = (key or "").strip()
        if not key:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM canonical_facts WHERE key=? "
                "ORDER BY valid_from_cycle, canonical_id",
                (key,),
            ).fetchall()
            return [dict(r) for r in rows]

    def review_canonical(self, key, status: str) -> bool:
        """Set review_status (e.g. reviewed / disputed) on the CURRENT row for
        `key`. Returns True if a current row was updated."""
        key = (key or "").strip()
        if not key:
            return False
        with self._lock:
            cur = self._conn.execute(
                "UPDATE canonical_facts SET review_status=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE key=? AND valid_until_cycle IS NULL",
                (status, key),
            )
            self._conn.commit()
            return cur.rowcount > 0
