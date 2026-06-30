"""store_batches.py — BatchMixin: semantic write-batch provenance + rollback.

Mixed into LatticeStore. A consolidation epoch / dream cycle opens a batch; every
fact written while it is open is stamped with batch_id (in store_facts'
add_or_reinforce_fact), so a bad generative run can be reviewed and rolled back as a
unit instead of by manual row cleanup. Relies on the composite for self._conn,
self._lock, self.get_cycle_counts; uses self._active_batch_id (set here, read in
store_facts via getattr so an un-batched write is simply NULL-stamped).

Scope: rollback removes the NEW facts a batch wrote. It does not restore rows that
the same dream cycle decayed or pruned (those are separate forgetting paths) — its
job is to undo bad WRITES, which is the durable-pollution risk."""

import logging

logger = logging.getLogger(__name__)


class BatchMixin:

    def begin_write_batch(self, phase, *, model=None, source_session=None,
                          config_hash=None, cycle=None):
        """Open a write batch; facts added until end_write_batch() are stamped with
        its id. Auto-closes any still-open batch first (self-healing). Returns batch_id."""
        if cycle is None:
            cycle = self.get_cycle_counts()[0]
        with self._lock:
            prior = getattr(self, "_active_batch_id", None)
            if prior is not None:
                self._finalize_batch(prior)
            bid = self._conn.execute(
                "INSERT INTO write_batches(phase, source_session, model, config_hash, "
                "created_cycle) VALUES (?, ?, ?, ?, ?)",
                (str(phase), source_session, model, config_hash, cycle),
            ).lastrowid
            self._conn.commit()
            self._active_batch_id = bid
            return bid

    def end_write_batch(self):
        """Close the active batch (record its write count; auto-clean if it wrote
        nothing). Idempotent / safe when no batch is open."""
        bid = getattr(self, "_active_batch_id", None)
        self._active_batch_id = None
        if bid is not None:
            with self._lock:
                self._finalize_batch(bid)
        return bid

    def _finalize_batch(self, bid):
        n = self._conn.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE batch_id=?", (bid,)).fetchone()[0]
        if n == 0:
            # an empty batch (a cycle that wrote nothing) leaves no provenance noise
            self._conn.execute(
                "DELETE FROM write_batches WHERE batch_id=? AND status='active'", (bid,))
        else:
            self._conn.execute(
                "UPDATE write_batches SET n_writes=?, "
                "status=CASE WHEN status='active' THEN 'closed' ELSE status END, "
                "closed_at=COALESCE(closed_at, CURRENT_TIMESTAMP) WHERE batch_id=?",
                (n, bid))
        self._conn.commit()

    def list_write_batches(self, limit=50):
        """Recent write batches (newest first), each with a LIVE fact count so a
        partially-pruned batch reads honestly. Read-only."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_id, phase, source_session, model, config_hash, created_cycle, "
                "n_writes, status, created_at, closed_at, rolled_back_at "
                "FROM write_batches ORDER BY batch_id DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["live_facts"] = self._conn.execute(
                    "SELECT COUNT(*) FROM semantic_facts WHERE batch_id=?",
                    (d["batch_id"],)).fetchone()[0]
                out.append(d)
            return out

    def get_batch_facts(self, batch_id, limit=200):
        """The facts a batch wrote (the diff surface for review). Read-only."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, category, tier, resonance_count, pinned "
                "FROM semantic_facts WHERE batch_id=? ORDER BY id LIMIT ?",
                (batch_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def rollback_write_batch(self, batch_id):
        """Delete the (non-pinned) facts a batch wrote and mark it rolled_back. Pinned
        facts are KEPT — a deliberate user lock overrides a batch rollback — and reported.
        The AFTER DELETE trigger cleans the vector/FTS/entity rows; relation rows cascade.
        Returns {batch_id, deleted, kept_pinned, status} or {error}."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM write_batches WHERE batch_id=?", (batch_id,)).fetchone()
            if row is None:
                return {"error": f"no such batch {batch_id}"}
            kept_pinned = self._conn.execute(
                "SELECT COUNT(*) FROM semantic_facts WHERE batch_id=? AND pinned=1",
                (batch_id,)).fetchone()[0]
            deleted = self._conn.execute(
                "DELETE FROM semantic_facts WHERE batch_id=? AND pinned=0",
                (batch_id,)).rowcount
            self._conn.execute(
                "UPDATE write_batches SET status='rolled_back', "
                "rolled_back_at=CURRENT_TIMESTAMP, n_writes=? WHERE batch_id=?",
                (kept_pinned, batch_id))
            self._conn.commit()
            return {"batch_id": batch_id, "deleted": deleted,
                    "kept_pinned": kept_pinned, "status": "rolled_back"}
