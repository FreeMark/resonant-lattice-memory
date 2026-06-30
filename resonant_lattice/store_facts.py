"""store_facts.py — FactsMixin: fact storage, dedup/reinforce, entity links,
single-fact + entity-scoped reads, resonance feedback.

Mixed into LatticeStore; uses self._conn/_lock/config attrs and the sibling
mixins' methods (self._extract_entities, etc.) via the composite."""

import logging
import re
import struct
from typing import List, Dict, Optional, Tuple

from store_common import serialize_vector, hrr, sqlite3

logger = logging.getLogger(__name__)


class FactsMixin:

    def _current_memory_cycle(self) -> int:
        """Current memory_cycle from the meta table — the logical clock used to
        stamp temporal fields (Phase 1). Single source of truth; never wall-clock.

        One indexed PK lookup. Acquires self._lock (an RLock, so this is safe both
        standalone and when nested inside an already-locked write path).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='memory_cycle'"
            ).fetchone()
            try:
                return int(row["value"]) if row else 0
            except (TypeError, ValueError):
                return 0

    # ====================== CORE FACT STORAGE ======================
    def _entities_compatible_for_merge(self, existing_id: int, new_entities) -> bool:
        """Guard the near-identity silent merge against ENTITY contamination.

        Two facts can be >= reinforce_threshold similar in EMBEDDING yet be about
        DIFFERENT entities — "Acme Corp's fee is 4050 cents" vs "Acme Inc's fee is
        9000 cents" (the embedder barely separates 'Corp' from 'Inc'). Folding the
        second into the first would silently conflate two companies AND discard the
        second amount — a critical failure for a money agent.

        Returns True only when it is safe to merge: the new fact names the SAME
        subject as the existing one (entity sets equal, or one a subset of the
        other — a re-statement with more/less detail). Disjoint or diverging entity
        sets => different subjects => do NOT merge (caller stores it separately).
        Conservative: if either side has no entities to disambiguate on, fall back
        to the legacy merge (preserves existing behaviour for entity-less facts).
        """
        new = {e.strip().lower() for e in (new_entities or []) if e and str(e).strip()}
        if not new:
            return True
        rows = self._conn.execute(
            "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE fe.fact_id = ?", (existing_id,)
        ).fetchall()
        existing = {r["name"].strip().lower() for r in rows if r["name"]}
        if not existing:
            return True
        return new <= existing or existing <= new

    def _specifics_compatible_for_merge(self, existing_id: int, new_content: str) -> bool:
        """Guard the near-identity silent merge against a VALUE UPDATE.

        Near-identical text that changes a hard NUMBER ("payment terms Net-30" ->
        "Net-60", "fee 4050 cents" -> "9000 cents") is a value update, not a
        reinforcement. Folding it into the existing row would silently KEEP THE
        STALE value and discard the new one. Returns False when the new content's
        multi-digit numbers diverge from the existing fact's (one isn't a subset
        of the other), so the update is stored separately instead of dropped.
        Conservative: no multi-digit numbers on either side => compatible (legacy).
        """
        def nums(t):
            return {re.sub(r"\D", "", m) for m in re.findall(r"\d[\d,]*", t or "")
                    if len(re.sub(r"\D", "", m)) >= 2}
        new_n = nums(new_content)
        if not new_n:
            return True
        row = self._conn.execute(
            "SELECT content FROM semantic_facts WHERE id = ?", (existing_id,)).fetchone()
        old_n = nums(row["content"] if row else "")
        if not old_n:
            return True
        return new_n <= old_n or old_n <= new_n

    def add_or_reinforce_fact(
        self,
        content: str,
        embedding: List[float],
        category: str = "general",
        source_session: str = "",
        hrr_vector: Optional["np.ndarray"] = None,
        entities: Optional[List[str]] = None,
        source_quote: Optional[str] = None,
        source_ref: Optional[str] = None,
        quote_status: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Semantic + exact match reinforcement with full HRR + entity graph support.

        source_quote / source_ref / quote_status (Phase D grounding + attestation)
        are recorded ONLY on a fresh INSERT. On reinforcement of an existing fact
        the first-seen provenance is kept — we don't overwrite the original
        supporting evidence with a later match's quote. All are optional with safe
        defaults, so existing callers and the (action, fact_id) return contract are
        unchanged.
        """
        content = content.strip()
        if not content:
            return "skipped", -1

        with self._lock:
            entities = entities or self._extract_entities(content)

            # 1. Semantic similarity check first (fast vector lookup).
            #    Fetch the top-1 neighbour (threshold 0.0) so the SAME lookup both
            #    gates the silent merge at near-identity (reinforce_threshold — a
            #    *changed* fact in the 0.78–0.95 band is stored separately so it can
            #    be flagged as a conflict, not dropped as a reinforcement) AND
            #    yields Phase-3 novelty (no extra query).
            top = self._find_semantic_match(embedding, threshold=0.0)
            if (top and top["similarity"] >= self.reinforce_threshold
                    and self._entities_compatible_for_merge(top["id"], entities)
                    and self._specifics_compatible_for_merge(top["id"], content)):
                # near-identical, same subject, same specifics -> safe to fold together
                self._reinforce_fact(top["id"])
                self._link_entities(top["id"], entities)
                return "reinforced (semantic)", top["id"]
            # Near-identical but NOT safe to merge -> INSERT a separate fact rather
            # than conflate: either the entity sets DIVERGE (different companies =
            # cross-entity contamination) or a hard NUMBER changed (a value update
            # that must not be dropped as a reinforcement of the stale value).

            # Phase 3 salience: novelty = 1 - top_similarity (fully novel when there
            # is no neighbour at all). A surprising fact imprints harder on first
            # contact; a near-duplicate barely moves. Bounded to [0,1]. NOTE: with
            # novelty_boost a fully-novel one-shot can clear promotion_threshold
            # even when base initial_resonance is below it (the intended effect —
            # see the initial_resonance<promotion_threshold warning in store.py).
            novelty = 1.0 - (top["similarity"] if top else 0.0)
            novelty = max(0.0, min(1.0, novelty))
            effective_initial = self.initial_resonance + (
                self.novelty_boost * novelty if self.novelty_enabled else 0.0
            )

            # 2. Exact string match fallback
            # Phase 1: stamp the logical clock at first INSERT. learned_at and
            # last_confirmed both start at the current memory_cycle.
            cur_cycle = self._current_memory_cycle()
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO semantic_facts
                    (content, category, tier, resonance_count, source_session, hrr_vector,
                     source_quote, source_ref, quote_status,
                     learned_at_cycle, last_confirmed_cycle, max_resonance_seen, batch_id)
                    VALUES (?, ?, 'short', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (content, category, effective_initial, source_session,
                     hrr.phases_to_bytes(hrr_vector)
                     if (hrr_vector is not None and hrr is not None) else None,
                     source_quote, source_ref, quote_status,
                     cur_cycle, cur_cycle, effective_initial,
                     getattr(self, "_active_batch_id", None)),
                )
                fact_id = cur.lastrowid
                # Store embedding (skipped in degraded mode — vec dim mismatch,
                # FTS-only fallback so the insert can't raise on a bad dim).
                if not self.degraded:
                    self._conn.execute(
                        "INSERT INTO semantic_vec (id, embedding) VALUES (?, ?)",
                        (fact_id, serialize_vector(embedding)),
                    )
                self._conn.commit()
                # FIX: previously fell through here → returned None (TypeError on
                # tuple-unpack at every call site) and never linked entities for
                # newly inserted facts.
                self._link_entities(fact_id, entities)
                return "added", fact_id

            except sqlite3.IntegrityError:
                # Exact duplicate → reinforce. If the integrity error wasn't the
                # content-unique collision, there's no row to reinforce; roll back
                # and skip rather than crash on row["id"].
                row = self._conn.execute(
                    "SELECT id FROM semantic_facts WHERE content = ?", (content,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    logger.debug("add_or_reinforce_fact: IntegrityError with no matching content row; skipped")
                    return "skipped", -1
                self._reinforce_fact(row["id"])
                self._link_entities(row["id"], entities)
                return "reinforced (exact)", row["id"]
            except Exception:
                # Any other DB error: roll back the half-applied transaction so
                # the shared connection stays usable for the next locked op.
                self._conn.rollback()
                raise


    def _find_semantic_match(self, embedding: List[float],
                             threshold: Optional[float] = None) -> Optional[Dict]:
        """Fast top-1 semantic lookup.

        semantic_vec uses distance_metric=cosine, so (1.0 - distance) is true
        cosine similarity. `threshold` defaults to self.similarity_threshold
        (recall/dedup); callers gating a silent *merge* pass self.reinforce_threshold
        so only near-identical facts are folded together.
        """
        if not embedding or self.degraded:
            return None
        cutoff = self.similarity_threshold if threshold is None else threshold
        vec_bytes = serialize_vector(embedding)
        # Phase 1b: superseded facts are retired history — never a dedup/reinforce
        # target. If the nearest neighbour is superseded the gate returns no match,
        # so a re-asserted belief enters as a fresh live fact (and may re-conflict
        # with the current winner) rather than reinforcing a retired row.
        row = self._conn.execute(
            """
            SELECT f.id, (1.0 - v.distance) as similarity
            FROM semantic_vec v
            JOIN semantic_facts f ON f.id = v.id
            WHERE v.embedding MATCH ? AND k = 1
              AND f.tier != 'superseded'
            ORDER BY v.distance LIMIT 1
            """,
            (vec_bytes,),
        ).fetchone()
        if row and row["similarity"] >= cutoff:
            return dict(row)
        return None


    def _reinforce_fact(self, fact_id: int) -> None:
        """Long-Term Potentiation.

        Phase 1: also refreshes last_confirmed_cycle to the current memory_cycle —
        a reinforcement (semantic or exact match) is a confirmation, so this is
        the freshness signal Phase 2 reads. Legacy rows with NULL temporal stamps
        get back-stamped here on their next reinforcement.
        """
        cur_cycle = self._current_memory_cycle()
        self._conn.execute(
            """
            UPDATE semantic_facts
            SET resonance_count = resonance_count + 1,
                last_confirmed_cycle = ?,
                max_resonance_seen = MAX(COALESCE(max_resonance_seen, 0), resonance_count + 1),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (cur_cycle, fact_id),
        )
        self._conn.commit()


    def _link_entities(self, fact_id: int, entities: List[str]) -> None:
        """Link extracted entities (fast, no LLM).

        Batched to avoid the previous N+1 SELECT/INSERT + per-call commit:
        one bulk INSERT OR IGNORE to ensure rows exist, one SELECT to resolve
        ids, one executemany to link. NOTE: this method COMMITS on completion
        (callers must already hold self._lock). It is therefore not safe to
        nest inside a larger transaction you intend to roll back — the commit
        here will land the entity rows regardless.
        """
        if not entities:
            return
        names = list(dict.fromkeys(entities))  # dedup, preserve order
        self._conn.executemany(
            "INSERT OR IGNORE INTO entities (name) VALUES (?)",
            [(n,) for n in names],
        )
        placeholders = ",".join("?" * len(names))
        rows = self._conn.execute(
            f"SELECT entity_id, name FROM entities WHERE name IN ({placeholders})",
            tuple(names),
        ).fetchall()
        before = self._conn.total_changes
        self._conn.executemany(
            "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
            [(fact_id, r["entity_id"]) for r in rows],
        )
        # Blind-tier: if a genuinely NEW (fact, entity) link landed, the fact's AEAD entity set
        # in semantic_he_entities is now stale — flag it so _blind_reconcile re-mirrors it (the
        # entity set is the one mutable blind source; embedding/HRR are immutable). The
        # total_changes delta catches only real inserts, so an idempotent re-link of the same
        # entities on exact-match reinforcement does NOT trigger a wasted re-encrypt.
        if self._conn.total_changes != before:
            self._conn.execute(
                "UPDATE semantic_facts SET entities_dirty = 1 WHERE id = ?", (fact_id,)
            )
        self._conn.commit()

            
    def get_fact(self, fact_id: int) -> Optional[Dict]:
        """Fetch a single fact by exact ID — direct lookup, no neighbours.

        Returns the row as a dict, or None if no fact has that id. This is the
        mechanical counterpart to semantic search: callers needing to confirm a
        specific stored row use this instead of abusing search() (whose
        neighbour rows invite confabulation). Read-only; under the lock.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, content, category, tier, resonance_count,
                       conflict_group_id, source_session, source_quote, source_ref,
                       quote_status, learned_at_cycle, last_confirmed_cycle,
                       superseded_by, superseded_at_cycle, max_resonance_seen, pinned
                FROM semantic_facts WHERE id = ?
                """,
                (fact_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_fact_history(self, fact_id: int) -> Optional[Dict]:
        """Walk a fact's supersedion lineage (Phase 1b). Read-only, under the lock.

        Returns the fact's temporal row plus:
          - 'superseded_by_chain': the forward chain of facts that replaced it,
            following superseded_by until NULL — the path toward current belief
            (each hop is a later winner that itself lost a subsequent conflict).
          - 'replaced': the facts this one superseded (its predecessors), newest
            first.
        Returns None if the id doesn't exist. Cycle-guarded (a self-referential or
        looping chain can't spin) and depth-capped at 50 hops.
        """
        with self._lock:
            base = self._conn.execute(
                """
                SELECT id, content, category, tier, resonance_count,
                       learned_at_cycle, last_confirmed_cycle,
                       superseded_by, superseded_at_cycle
                FROM semantic_facts WHERE id = ?
                """,
                (fact_id,),
            ).fetchone()
            if base is None:
                return None
            chain = []
            seen = {fact_id}
            nxt = base["superseded_by"]
            while nxt is not None and nxt not in seen and len(chain) < 50:
                row = self._conn.execute(
                    """
                    SELECT id, content, tier, resonance_count,
                           superseded_by, superseded_at_cycle
                    FROM semantic_facts WHERE id = ?
                    """,
                    (nxt,),
                ).fetchone()
                if row is None:
                    break
                seen.add(nxt)
                chain.append(dict(row))
                nxt = row["superseded_by"]
            replaced = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT id, content, tier, superseded_at_cycle
                    FROM semantic_facts WHERE superseded_by = ?
                    ORDER BY superseded_at_cycle DESC, id DESC
                    """,
                    (fact_id,),
                ).fetchall()
            ]
            result = dict(base)
            result["superseded_by_chain"] = chain
            result["replaced"] = replaced
            return result

    def set_pinned(self, fact_id: int, pinned: bool = True) -> bool:
        """Pin/unpin a fact as identity-level / never-forget (A5, P4a). Returns True if a row matched.

        A pinned fact is excluded from EVERY forgetting path (cycle decay, staleness decay,
        dormant-prune, long-tier-cap eviction) — the one kind of memory the system will not let
        fade. Pinning is purely PROTECTIVE: it does not touch resonance_count, so it can never be
        used to inflate a fact into runaway immortality; the system still owns decay/forget for
        everything unpinned. Idempotent (re-pinning a pinned fact returns True, changes nothing).
        Lock-guarded. The inverse of the (admin-only) remove path: the agent retires a wrong fact
        via unhelpful feedback (fades to dormant, recoverable) and protects a vital one via pin.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE semantic_facts SET pinned = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (1 if pinned else 0, fact_id),
            )
            self._conn.commit()
            return (cur.rowcount or 0) > 0

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a single fact by exact ID. Returns True if a row was removed.

        The semantic_facts_ad trigger cascades cleanup of semantic_vec, the FTS
        index, and fact_entities; entity rows left with no remaining link are
        swept separately by gc_orphan_entities(). Lock-guarded; idempotent
        (a second call for the same id returns False).
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM semantic_facts WHERE id = ?", (fact_id,)
            )
            self._conn.commit()
            return (cur.rowcount or 0) > 0


    # ====================== BLIND-TIER READ-BACK (reconciliation) ======================
    # Read a fact's plaintext embedding / HRR back out of the store so the blind-tier
    # reconciliation (provider _blind_reconcile, roadmap §14 6a) can mirror facts created
    # outside the consolidation hook (abstraction/gist/procedural/builtin) + backfill an
    # existing store — WITHOUT re-embedding via Ollama. The stored float32 round-trips exactly.
    def get_fact_embedding(self, fact_id: int) -> Optional[List[float]]:
        """Read a fact's embedding back from semantic_vec as a float list (None if absent or in
        degraded mode). Mirrors serialize_vector's float32 packing; the round-trip is exact."""
        if getattr(self, "degraded", False):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM semantic_vec WHERE id = ?", (int(fact_id),)
            ).fetchone()
        if not row or row["embedding"] is None:
            return None
        blob = bytes(row["embedding"])
        return list(struct.unpack(f"{len(blob) // 4}f", blob))

    def get_fact_hrr_phases(self, fact_id: int):
        """Read a fact's HRR phase vector back from semantic_facts.hrr_vector (None if absent or
        dim-mismatched — reuses _phases_from_blob, which guards dimension drift)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT hrr_vector FROM semantic_facts WHERE id = ?", (int(fact_id),)
            ).fetchone()
        if not row or row["hrr_vector"] is None:
            return None
        return self._phases_from_blob(row["hrr_vector"])


    # ====================== ENTITY GRAPH QUERY METHODS ======================
    def get_facts_for_entity(self, entity_name: str, limit: int = 20,
                             tier: Optional[str] = None,
                             category: Optional[str] = None) -> List[Dict]:
        """Return facts linked to a specific entity, ranked by resonance.

        Optional `category` (e.g. 'tool_action') narrows to one fact class —
        used by get_tool_history().
        """
        with self._lock:
            params = [entity_name.lower()]
            query = """
                SELECT f.id, f.content, f.category, f.tier, f.resonance_count,
                       f.conflict_group_id, f.source_session
                FROM semantic_facts f
                JOIN fact_entities fe ON fe.fact_id = f.id
                JOIN entities e ON e.entity_id = fe.entity_id
                WHERE LOWER(e.name) = ?
                  AND f.tier != 'superseded'
            """
            if tier:
                query += " AND f.tier = ?"
                params.append(tier)
            if category:
                query += " AND f.category = ?"
                params.append(category)
            query += " ORDER BY f.resonance_count DESC, f.updated_at DESC LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

                

    def adjust_resonance(self, fact_id: int, delta: int) -> bool:
        """Hebbian feedback from fact_feedback tool."""
        with self._lock:
            row = self._conn.execute(
                "SELECT resonance_count FROM semantic_facts WHERE id = ?", (fact_id,)
            ).fetchone()
            if not row:
                return False
            new_res = max(0, row["resonance_count"] + delta)
            # Phase 3: keep the peak high-water mark current — positive feedback is
            # a strong 'this was important' signal P4 leans on.
            self._conn.execute(
                "UPDATE semantic_facts SET resonance_count = ?, "
                "max_resonance_seen = MAX(COALESCE(max_resonance_seen, 0), ?), "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_res, new_res, fact_id),
            )
            self._conn.commit()
            return True


    def reinforce_on_recall(self, fact_ids: List[int], bump: float) -> int:
        """Small resonance bump for facts that were actually recalled/used.

        Hebbian 'use it or lose it': retrieval is a weak positive signal, so the
        bump is much smaller than explicit feedback (+2). The provider gates this
        to at most once per fact per dream-cycle window so frequently-matching
        facts don't run away. Normal decay still applies. Returns rows updated.
        """
        if not fact_ids or bump == 0:
            return 0
        with self._lock:
            placeholders = ",".join("?" * len(fact_ids))
            # Soft ceiling: recall + tool-success bumps must not let a hot fact
            # grow without bound (which would make it effectively immortal and
            # immune to decay). Saturate at a high cap instead.
            cur = self._conn.execute(
                f"UPDATE semantic_facts "
                f"SET resonance_count = MIN(resonance_count + ?, 50.0) "
                f"WHERE id IN ({placeholders})",
                (bump, *fact_ids),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def seed_procedural_facts(self, items: List[Dict], current_cycle: int = 0,
                             durable_resonance: float = 10.0) -> int:
        """Seed durable procedural / guardrail facts (e.g. tool-usage rules and 'how NOT to use it'
        guardrails) so the agent is grounded from DAY ONE — before it has failed enough to learn them
        (P3e). ``items`` = [{"content", "embedding", "entities"?}], pre-embedded by the caller (the
        store calls no Ollama). Inserted as category='procedural', tier='long', high resonance so a
        guardrail does NOT decay away before it is learned; idempotent (skips a content already
        present), so re-seeding on every startup is safe. Returns the count newly inserted.

        Phrase guardrails POSITIVELY where possible ('always require human approval') rather than
        naming the forbidden capability ('never auto_approve') — the P3f judge showed the negative
        form primes small models to do the very thing it forbids."""
        n = 0
        with self._lock:
            for it in items:
                content = (it.get("content") or "").strip()
                emb = it.get("embedding")
                if not content or not emb:
                    continue
                if self._conn.execute("SELECT 1 FROM semantic_facts WHERE content = ?",
                                      (content,)).fetchone():
                    continue   # idempotent: already seeded / known
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO semantic_facts (content, category, tier, resonance_count, "
                    "source_session, learned_at_cycle, last_confirmed_cycle, max_resonance_seen) "
                    "VALUES (?, 'procedural', 'long', ?, 'seed', ?, ?, ?)",
                    (content, float(durable_resonance), current_cycle, current_cycle,
                     float(durable_resonance)))
                fid = cur.lastrowid
                if not fid:
                    continue
                if not self.degraded:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO semantic_vec (id, embedding) VALUES (?, ?)",
                        (fid, serialize_vector(emb)))
                ents = it.get("entities") or []
                if ents:
                    self._link_entities(fid, ents)
                n += 1
            self._conn.commit()
        return n
