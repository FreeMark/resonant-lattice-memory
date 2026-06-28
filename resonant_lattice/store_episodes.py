"""store_episodes.py — EpisodesMixin: conversational (L1) + procedural
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

        Uses GROUP BY + MAX(id) to identify the most recent sessions —
        avoids the undefined behavior of ORDER BY inside SELECT DISTINCT.

        `max_rows` (0 = unlimited) additionally caps the TOTAL episode count,
        deleting oldest-first. Guards against a single never-ending session
        growing the table without bound (the session filter alone can't).
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
                    """,
                    (max_rows,)
                )
            self._conn.commit()


    def add_episode(self, session_id: str, role: str, content: str) -> None:
        """Append one conversational turn to the episodic (L1) layer."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            self._conn.commit()

            
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
                return -1  # duplicate call_id — replay rejected by DB
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
