"""store_episodes.py - EpisodesMixin: conversational (L1) + procedural
(tool) episode logs and their bounded pruning.

Mixed into LatticeStore; relies on the composite for self._conn/_lock."""

import logging
import json
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class EpisodesMixin:

    # Convenience wrapper for the lattice_store `tool_history` action.
    # Returns RAW tool episodes (the procedural L1 log). Distilled procedural
    # FACTS use category='procedural' and surface through normal recall /
    # facts_about_entity instead.
    def get_tool_history(self, tool_name: str, limit: int = 10, tier: Optional[str] = None) -> List[Dict]:
        """Return recent RAW tool episodes for a tool (the procedural log).

        `tier` is accepted for backward-compatibility but ignored: raw episodes
        are not tiered. Procedural FACTS distilled from these episodes are tiered
        and are retrieved via normal semantic search.
        """
        if not tool_name:
            return []
        return self.get_recent_tool_episodes(tool_name=tool_name, limit=limit)

            
    def prune_episodes(self, keep_sessions: int = 20, max_rows: int = 0) -> None:
        """Delete episode rows outside the N most recent sessions.

        Uses GROUP BY + MAX(id) to identify the most recent sessions -
        avoids the undefined behavior of ORDER BY inside SELECT DISTINCT.

        `max_rows` (0 = unlimited) additionally caps the TOTAL episode count,
        deleting oldest-first. Guards against a single never-ending session
        growing the table without bound (the session filter alone can't).

        CONSOLIDATION-DEBT EXEMPTION: sessions flagged as open debt (substantial
        episodes, zero banked facts - see flag_consolidation_debt) are never
        pruned by either clause. Their episodes are the only remaining evidence
        of an undigested experience; deleting them made the TanStack loss
        unrecoverable. The exemption lifts as soon as the debt settles.
        """
        with self._lock:
            self._conn.execute(
                """
                DELETE FROM episodes
                WHERE session_id NOT IN (
                    SELECT session_id
                    FROM episodes
                    GROUP BY session_id
                    ORDER BY MAX(id) DESC
                    LIMIT ?
                )
                AND session_id NOT IN (
                    SELECT session_id FROM consolidation_debt WHERE settled IS NULL
                )
                """,
                (keep_sessions,)
            )
            if max_rows > 0:
                self._conn.execute(
                    """
                    DELETE FROM episodes
                    WHERE id NOT IN (
                        SELECT id FROM episodes ORDER BY id DESC LIMIT ?
                    )
                    AND session_id NOT IN (
                        SELECT session_id FROM consolidation_debt WHERE settled IS NULL
                    )
                    """,
                    (max_rows,)
                )
            self._conn.commit()


    # ================== CONSOLIDATION DEBT (undigested sessions) ==================
    # A session with substantial episodes and ZERO facts born from it is an
    # experience the substrate failed to digest. The ledger lives here because
    # debt is an episode-lifecycle concern: it is flagged from the episode log,
    # it exempts episodes from pruning, and it settles when facts finally land.

    # A debt needs at least this many episode rows (2 full user+assistant turns);
    # smaller sessions are legitimately fact-free smalltalk, not failures.
    DEBT_MIN_EPISODES = 4

    def flag_consolidation_debt(self, current_cycle: int,
                                exclude_session: str = "",
                                min_episodes: int = 0) -> int:
        """Flag sessions with >= min_episodes episode rows and zero born facts.

        Pure SQL, idempotent (a session is flagged once, ever). ``exclude_session``
        keeps the CURRENT session out - its consolidation may simply not have run
        yet. A concurrent process's live session can still be flagged early; that
        false positive self-heals because the retry path re-checks born facts
        before spending any LLM call and settles it 'organic'. Returns rows added.
        """
        m = int(min_episodes) if min_episodes > 0 else self.DEBT_MIN_EPISODES
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO consolidation_debt
                    (session_id, episode_rows, first_flagged_cycle)
                SELECT e.session_id, COUNT(*), ?
                FROM episodes e
                WHERE e.session_id != ?
                  AND e.session_id NOT IN
                      (SELECT session_id FROM consolidation_debt)
                  AND e.session_id NOT IN
                      (SELECT DISTINCT source_session FROM semantic_facts
                       WHERE source_session IS NOT NULL)
                GROUP BY e.session_id
                HAVING COUNT(*) >= ?
                """,
                (int(current_cycle), exclude_session or "", m),
            )
            self._conn.commit()
            added = cur.rowcount or 0
            if added:
                logger.info("Consolidation debt: flagged %d undigested session(s)", added)
            return added

    def get_open_consolidation_debts(self, before_cycle: int, limit: int = 1) -> List[Dict]:
        """Open debts flagged BEFORE the given dream cycle, oldest first.

        The one-cycle seasoning delay means a session flagged mid-flight by a
        concurrent process gets a full dream cycle to consolidate on its own
        before we touch it."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id, episode_rows, attempts, first_flagged_cycle
                FROM consolidation_debt
                WHERE settled IS NULL AND first_flagged_cycle < ?
                ORDER BY first_flagged_cycle ASC
                LIMIT ?
                """,
                (int(before_cycle), int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]

    def session_born_fact_count(self, session_id: str) -> int:
        """Facts whose source_session is this session (the debt receipt check)."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM semantic_facts WHERE source_session = ?",
                (session_id,),
            ).fetchone()[0]

    def settle_consolidation_debt(self, session_id: str, outcome: str,
                                  current_cycle: int = 0) -> None:
        """Close a debt: 'recovered' (retry banked facts), 'organic' (facts
        appeared without us), or 'exhausted' (attempts used up, still zero)."""
        with self._lock:
            self._conn.execute(
                "UPDATE consolidation_debt SET settled = ?, last_attempt_cycle = ? "
                "WHERE session_id = ?",
                (outcome, int(current_cycle), session_id),
            )
            self._conn.commit()

    def record_debt_attempt(self, session_id: str, current_cycle: int) -> int:
        """Increment a debt's retry counter; returns the new attempt count."""
        with self._lock:
            self._conn.execute(
                "UPDATE consolidation_debt SET attempts = attempts + 1, "
                "last_attempt_cycle = ? WHERE session_id = ?",
                (int(current_cycle), session_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT attempts FROM consolidation_debt WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row[0] if row else 0

    def consolidation_debt_counts(self) -> Dict[str, int]:
        """Health-snapshot rollup: open / recovered / organic / exhausted."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(settled, 'open') AS state, COUNT(*) "
                "FROM consolidation_debt GROUP BY state"
            ).fetchall()
            return {r[0]: r[1] for r in rows}


    def add_episode(self, session_id: str, role: str, content: str) -> None:
        """Append one conversational turn to the episodic (L1) layer."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            self._conn.commit()

    # ---- session_sources: transient citation evidence -----------------------
    #
    # These three carry the url bridge between sync_turn (which sees the raw tool
    # messages) and consolidation (which sees only a session_id). See
    # SchemaMixin._migrate_add_session_sources for why the page body is staged, and
    # source_index.py for what is done with it. The table is TRANSIENT: nothing in
    # the substrate depends on a row surviving, so pruning is always safe.

    def add_session_sources(self, session_id: str,
                            sources: List[Dict[str, str]]) -> int:
        """Stage retrieved (url, title, body) triples. Returns rows newly inserted.

        INSERT OR IGNORE against the UNIQUE (session_id, url) index, because hermes
        replays message history on restart and the same page will arrive repeatedly.
        """
        if not session_id or not sources:
            return 0
        rows = [(session_id, s.get("url") or "", s.get("title") or "",
                 s.get("body") or "")
                for s in sources if (s.get("url") or "").strip()]
        if not rows:
            return 0
        with self._lock:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO session_sources "
                "(session_id, url, title, body) VALUES (?, ?, ?, ?)", rows,
            )
            self._conn.commit()
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def get_session_sources(self, session_id: str, limit: int = 200,
                            with_body: bool = False) -> List[Dict[str, str]]:
        """Retrieved sources for a session, oldest first.

        `with_body` is off by default so building the prompt index -- which needs
        only url and title -- never pulls megabytes of page text into memory.
        """
        if not session_id:
            return []
        cols = "url, title, body" if with_body else "url, title, '' AS body"
        with self._lock:
            rows = self._conn.execute(
                "SELECT %s FROM session_sources WHERE session_id = ? "
                "ORDER BY id LIMIT ?" % cols, (session_id, int(limit)),
            ).fetchall()
        return [{"url": r[0], "title": r[1] or "", "body": r[2] or ""}
                for r in rows]

    def prune_session_sources(self, older_than_hours: float = 48.0) -> int:
        """Drop staged evidence past its usefulness. Returns rows removed.

        48h default: long enough that a slow or retried block can still verify its
        own citations, short enough that page text does not accumulate. Nothing
        references these rows once a session has consolidated.
        """
        # Sign is carried by %+f, not by a literal '-'. Writing "-%f" turns a
        # negative argument into "--1.000000 hours", which sqlite rejects as a
        # modifier and silently applies NOTHING -- the delete then reports 0 rows
        # and looks like "there was nothing to prune". A negative value is
        # legitimate and useful: it sets the cutoff in the FUTURE, which is how a
        # caller (or a test) force-clears the staging table.
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM session_sources WHERE created_at < "
                "datetime('now', ?)",
                ("%+f hours" % (-float(older_than_hours)),),
            )
            self._conn.commit()
            return cur.rowcount or 0

            
    def add_turn(self, session_id: str, user_content: str, assistant_content: str) -> None:
        """Append a full user+assistant turn atomically (single lock + commit).

        Inserting both rows under one lock prevents a fast next turn from landing
        its 'user' row between this turn's 'user' and 'assistant' rows, which
        would scramble the transcript the consolidation LLM reads.
        """
        with self._lock:
            self._conn.executemany(
                "INSERT INTO episodes (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", user_content),
                    (session_id, "assistant", assistant_content),
                ],
            )
            self._conn.commit()

            
    # ====================== PROCEDURAL (TOOL) EPISODE LOG ======================
    def add_tool_episode(self, session_id: str, tool_name: str, arguments: Any,
                         result: str = "", success: bool = False,
                         memory_cycle: int = 0,
                         call_id: Optional[str] = None) -> int:
        """Append one raw tool-invocation event to the procedural log.

        Episodic, not semantic: no dedup, no HRR, no per-event resonance. These
        accumulate and are later generalized into reusable 'procedural' facts by
        distill_procedural_facts(). Returns the new row id, or -1 if the row was
        suppressed as a duplicate call_id (restart replay of message history).
        """
        try:
            arg_str = json.dumps(arguments, ensure_ascii=False)[:1000]
        except Exception:
            arg_str = str(arguments)[:1000]
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO tool_episodes
                    (session_id, tool_name, arguments, result, success,
                     memory_cycle, call_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, (tool_name or "unknown").lower(), arg_str,
                 (result or "")[:2000], 1 if success else 0, int(memory_cycle),
                 call_id),
            )
            self._conn.commit()
            if (cur.rowcount or 0) == 0:
                return -1  # duplicate call_id - replay rejected by DB
            return cur.lastrowid


    def get_recent_tool_episodes(self, tool_name: Optional[str] = None,
                                 limit: int = 20) -> List[Dict]:
        """Return recent raw tool episodes, optionally filtered by tool name."""
        with self._lock:
            base = (
                "SELECT id, session_id, tool_name, arguments, result, success, "
                "memory_cycle, created_at FROM tool_episodes"
            )
            if tool_name:
                rows = self._conn.execute(
                    base + " WHERE tool_name = ? ORDER BY id DESC LIMIT ?",
                    (tool_name.lower(), limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    base + " ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["success"] = bool(d.get("success"))
                out.append(d)
            return out


    def session_tool_names(self, session_id: str) -> set:
        """Distinct tool names this session invoked - its tool profile.

        Used by synthesis detection (zero web tools + lattice reads = reflection).
        lattice_store itself is deliberately absent from tool_episodes; its read
        signal travels in-process via provider._lattice_read_sessions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT tool_name FROM tool_episodes WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return {r[0] for r in rows}


    def prune_tool_episodes(self, keep: int = 500) -> int:
        """Keep only the most recent `keep` tool episodes (bounded log)."""
        if keep <= 0:
            return 0
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM tool_episodes
                WHERE id NOT IN (
                    SELECT id FROM tool_episodes ORDER BY id DESC LIMIT ?
                )
                """,
                (keep,),
            )
            self._conn.commit()
            removed = cur.rowcount or 0
            if removed:
                logger.debug("Pruned %d old tool episodes (keep=%d)", removed, keep)
            return removed


    def get_recent_episodes(self, limit: int = 10, session_id: Optional[str] = None) -> List[Dict]:
        with self._lock:
            query = "SELECT role, content FROM episodes"
            params: list = []
            if session_id:
                query += " WHERE session_id = ?"
                params.append(session_id)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in reversed(rows)]
