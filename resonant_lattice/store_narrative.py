"""store_narrative.py — NarrativeMixin: Phase-8 autobiographical layer.

Mixed into LatticeStore; uses self._conn/_lock and sibling methods
(self.get_recent_episodes, self._clean_llm_json) via the composite.

Episodes are L1/ephemeral (pruned by the session window) and semantic facts are
atomic — neither preserves the THREAD of what happened across sessions. This layer
stores a durable, bounded, one-paragraph gist per session in the separate
session_summaries table, generated at session end. It is explicitly a remembered
SUMMARY (never verbatim), so it surfaces as "recent history" context without being
mistaken for an exact quote. Reuses the session-end consolidation path + the
gisting discipline from Phase 4.
"""

import json
import logging
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_SUMMARY_CHARS = 1500


class NarrativeMixin:

    def add_session_summary(self, session_id: str, summary: str,
                            started_cycle: Optional[int] = None,
                            ended_cycle: Optional[int] = None,
                            created_cycle: Optional[int] = None,
                            keep: int = 30) -> Optional[int]:
        """Store one session narrative summary, then bound the table to `keep`.

        Lock-guarded. Trims + length-caps the summary; skips empty. Returns the new
        summary_id (or None if the summary was empty). Pruning to the most recent
        `keep` happens in the same lock so the autobiographical log never grows
        unbounded.
        """
        if not summary or not summary.strip():
            return None
        summary = summary.strip()[:_MAX_SUMMARY_CHARS]
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO session_summaries
                    (session_id, summary, started_cycle, ended_cycle, created_cycle)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, summary, started_cycle, ended_cycle, created_cycle),
            )
            new_id = cur.lastrowid
            self._conn.commit()
        if keep and keep > 0:
            self.prune_session_summaries(keep)
        return new_id

    def get_recent_narrative(self, limit: int = 30,
                             chronological: bool = True) -> List[Dict]:
        """Return the most recent session summaries.

        Fetches the newest `limit` rows; with chronological=True returns them
        oldest→newest so they read as a story thread (for the system-prompt
        "recent history" block). Read-only.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT summary_id, session_id, summary,
                       started_cycle, ended_cycle, created_cycle
                FROM session_summaries
                ORDER BY COALESCE(created_cycle, 0) DESC, summary_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        rows = [dict(r) for r in rows]
        return list(reversed(rows)) if chronological else rows

    def prune_session_summaries(self, keep: int) -> int:
        """Keep only the most recent `keep` summaries; delete the rest. Returns the
        number removed. Bounded autobiographical history (oldest fade first)."""
        if keep is None or keep <= 0:
            return 0
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM session_summaries
                WHERE summary_id NOT IN (
                    SELECT summary_id FROM session_summaries
                    ORDER BY COALESCE(created_cycle, 0) DESC, summary_id DESC
                    LIMIT ?
                )
                """,
                (keep,),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def summarize_session(self, reason_model: str, ollama_endpoint: str,
                          session_id: str, *, prompt: Optional[str] = None,
                          started_cycle: Optional[int] = None,
                          ended_cycle: Optional[int] = None,
                          created_cycle: Optional[int] = None,
                          keep: int = 30, min_episodes: int = 2,
                          max_episodes: int = 40) -> Optional[int]:
        """Generate + store a one-paragraph narrative gist of a session (Phase 8).

        Gathers the session's recent episodes (locked), asks the reasoning model for
        a short autobiographical summary (unlocked), then stores it (locked) +
        bounds the table. Mirrors the consolidate_before_prune structure. Returns the
        new summary_id, or None when there is too little to summarise or the LLM
        call fails (non-fatal — never blocks session shutdown).
        """
        episodes = self.get_recent_episodes(limit=max_episodes, session_id=session_id)
        if not episodes or len(episodes) < min_episodes:
            return None
        transcript = "\n".join(
            f"{e['role'].upper()}: {e['content']}" for e in episodes
        )
        base_prompt = prompt or (
            "Summarise the session below as ONE short paragraph of durable "
            "autobiographical memory — what the user and assistant worked on and "
            "decided together, the kind of thing worth remembering next session. "
            "Frame it as a remembered summary, not a transcript; keep only the "
            "throughline, drop turn-by-turn detail; never invent anything not in "
            "the log. Output ONLY the paragraph, no preamble."
        )
        final_prompt = f"{base_prompt}\n\nSESSION LOG:\n{transcript}\n\nSUMMARY:"
        try:
            payload = {"model": reason_model, "prompt": final_prompt,
                       "stream": False, "options": {"temperature": 0.3}}
            req = urllib.request.Request(
                f"{ollama_endpoint}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300.0) as response:
                raw = json.loads(response.read().decode("utf-8")).get("response", "")
        except Exception as e:
            logger.debug("Session summarisation LLM call failed (non-fatal): %s", e)
            return None
        # Reuse the shared cleaner to strip <think> blocks / code fences, then take
        # the prose as-is (this is freeform narrative, not JSON).
        summary = self._clean_llm_json(raw).strip()
        if not summary:
            return None
        return self.add_session_summary(
            session_id, summary, started_cycle=started_cycle,
            ended_cycle=ended_cycle, created_cycle=created_cycle, keep=keep,
        )
