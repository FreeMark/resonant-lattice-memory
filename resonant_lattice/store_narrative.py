"""store_narrative.py - NarrativeMixin: Phase-8 autobiographical layer.

Mixed into LatticeStore; uses self._conn/_lock and sibling methods
(self.get_recent_episodes, self._clean_llm_json) via the composite.

Episodes are L1/ephemeral (pruned by the session window) and semantic facts are
atomic - neither preserves the THREAD of what happened across sessions. This layer
stores a durable, bounded, one-paragraph gist per session in the separate
session_summaries table, generated at session end. It is explicitly a remembered
SUMMARY (never verbatim), so it surfaces as "recent history" context without being
mistaken for an exact quote. Reuses the session-end consolidation path + the
gisting discipline from Phase 4.
"""

import json
import logging
import re
import unicodedata
import urllib.request
from reason_gate import reason_slot
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_SUMMARY_CHARS = 2200  # headroom for grok's structured arc (throughline/decisions/open/closed);
#                            hermes single-paragraph narratives sit well under this, so raising the
#                            cap only lets the richer form survive - it never truncates what worked.

# Non-ASCII punctuation the reasoning model tends to emit (em/en dashes, smart quotes,
# ellipsis, nbsp) -> plain ASCII. Applied to every stored narrative so the
# autobiographical surface stays ASCII-only regardless of the prompt or model (standing
# operator pin). Cosmetic-only: it never changes the meaning of a summary.
_ASCII_MAP = {
    "—": " - ", "–": " - ", "‒": " - ", "―": " - ",  # em/en/figure/bar dash
    "‑": "-",                                                        # non-breaking hyphen
    "‘": "'", "’": "'", "‚": "'", "‛": "'",           # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',           # double quotes
    "…": "...",                                                      # ellipsis
    " ": " ", " ": " ", " ": " ",                          # non-breaking spaces
}


def _ascii_sanitize(text: str) -> str:
    """Return an ASCII-only rendering of `text`: map the common non-ASCII punctuation to
    readable ASCII (dashes, quotes, ellipsis, nbsp), then fold accents and DROP anything
    still non-ASCII (arrows, bullets, symbols, CJK) so a stored narrative is guaranteed
    ASCII regardless of what the model emits. Newlines preserved; the ' - ' dash
    substitution's double spaces are collapsed."""
    if not text:
        return text
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    # NFKD folds accents (e-acute -> e) and compatibility forms; ascii/ignore then drops any
    # remaining non-ASCII the map above did not name (e.g. a right-arrow granite emits).
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[ \t]{2,}", " ", text)


# P1: the structured-narrative prompt. Same content discipline as the freeform Rivernest
# prompt (ground in the log, ASCII, resumable handles, current-status-not-old-gaps), but
# asks for a parseable JSON object so the arc can be stored in typed columns.
_DEFAULT_STRUCTURED_PROMPT = (
    "You are writing a remembered session arc for an AI agent's NEXT wake, not a diary. "
    "Ground every item in the SESSION LOG; never invent. Return ONLY a JSON object with keys:\n"
    "  \"throughline\": one sentence naming what the session was about.\n"
    "  \"decisions\": array of locked decisions / standing rules established (at most 5, one line each).\n"
    "  \"open_loops\": array of still-pending items (at most 4), each with a resumable handle "
    "(a profile name, file path, or lane id from the log) when load-bearing.\n"
    "  \"closed\": array of finished work (at most 5), each stated in past tense.\n"
    "  \"topics\": array of short topic tags the session touched.\n"
    "Use ONLY ASCII punctuation. If the session supersedes an earlier backlog or plan, reflect the "
    "CURRENT status, not the old gap list. Any array may be empty. Output ONLY the JSON object."
)


def _render_narrative_blob(throughline, decisions, open_loops, closed):
    """Deterministically render the structured fields into the human-readable `summary`
    text, so readers that only know the flat summary column still get the whole arc."""
    parts = []
    if throughline and str(throughline).strip():
        parts.append(str(throughline).strip())

    def _sec(label, items):
        items = [str(i).strip() for i in (items or []) if str(i).strip()]
        if items:
            parts.append(label + " " + "; ".join(items))

    _sec("Decisions:", decisions)
    _sec("Open loops:", open_loops)
    _sec("Closed:", closed)
    return _ascii_sanitize("  ".join(parts))


class NarrativeMixin:

    def add_session_summary(self, session_id: str, summary: str,
                            started_cycle: Optional[int] = None,
                            ended_cycle: Optional[int] = None,
                            created_cycle: Optional[int] = None,
                            keep: int = 30, *,
                            throughline: Optional[str] = None,
                            decisions: Optional[List] = None,
                            open_loops: Optional[List] = None,
                            closed: Optional[List] = None,
                            topics: Optional[List] = None) -> Optional[int]:
        """Store one session narrative summary, then bound the table to `keep`.

        Lock-guarded. ASCII-sanitizes + trims + length-caps the summary; skips empty.
        Returns the new summary_id (or None if empty). The optional P1 structured
        fields carry the machine-readable arc: `throughline` is TEXT; decisions /
        open_loops / closed / topics are lists serialised to JSON TEXT (each element
        sanitized). Pruning to the most recent `keep` happens in the same lock so the
        autobiographical log never grows unbounded.
        """
        if not summary or not summary.strip():
            return None
        summary = _ascii_sanitize(summary.strip())[:_MAX_SUMMARY_CHARS]

        def _jlist(v):
            if v is None:
                return None
            if isinstance(v, str):
                return _ascii_sanitize(v)
            items = [_ascii_sanitize(str(x)) for x in v if str(x).strip()]
            return json.dumps(items, ensure_ascii=True)

        th = _ascii_sanitize(str(throughline).strip()) if throughline else None
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO session_summaries
                    (session_id, summary, started_cycle, ended_cycle, created_cycle,
                     throughline, decisions, open_loops, closed, topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, summary, started_cycle, ended_cycle, created_cycle,
                 th, _jlist(decisions), _jlist(open_loops), _jlist(closed), _jlist(topics)),
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
                       started_cycle, ended_cycle, created_cycle,
                       throughline, decisions, open_loops, closed, topics,
                       COALESCE(historical, 0) AS historical
                FROM session_summaries
                ORDER BY COALESCE(created_cycle, 0) DESC, summary_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("decisions", "open_loops", "closed", "topics"):
                v = d.get(k)
                if v:
                    try:
                        d[k] = json.loads(v)
                    except Exception:
                        d[k] = [v]
                else:
                    d[k] = []
            out.append(d)
        return list(reversed(out)) if chronological else out

    def mark_prior_narratives_historical(self, keep_current: int = 1) -> int:
        """Recompute the historical flag from recency: the newest `keep_current` narratives
        are CURRENT (historical=0), every older one is historical=1 (temporal framing). This
        is a FULL recompute in both directions - a row that later becomes the newest is
        cleared back to current, not left stale (the one-directional set-only version left a
        previously-historical row historical even after it became newest). Returns the number
        of rows whose flag actually changed. Idempotent; same recency ordering as pruning.
        Callers that want the newest to read as 'current' (the grok projection) invoke this
        right after writing a new narrative; the default (hermes) path never does, so its
        behaviour is unchanged."""
        keep_current = max(0, keep_current)
        with self._lock:
            newest = (
                "SELECT summary_id FROM session_summaries "
                "ORDER BY COALESCE(created_cycle, 0) DESC, summary_id DESC LIMIT ?"
            )
            demote = self._conn.execute(
                "UPDATE session_summaries SET historical = 1 "
                "WHERE COALESCE(historical, 0) = 0 AND summary_id NOT IN (%s)" % newest,
                (keep_current,),
            ).rowcount or 0
            promote = self._conn.execute(
                "UPDATE session_summaries SET historical = 0 "
                "WHERE COALESCE(historical, 0) = 1 AND summary_id IN (%s)" % newest,
                (keep_current,),
            ).rowcount or 0
            self._conn.commit()
            return demote + promote

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
                          digest: Optional[str] = None,
                          structured: bool = False,
                          structured_prompt: Optional[str] = None,
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
        call fails (non-fatal - never blocks session shutdown).

        When `digest` is provided (the grok hierarchical-ingest path) it is used
        verbatim as the SESSION LOG body and both the episode gather and the
        `max_episodes` cap are skipped - the caller has already distilled the WHOLE
        session (per-window born facts + open loops + a head/mid/tail spine), so the
        model sees all of it instead of only the tail-`max_episodes` episodes that a
        long multi-window ingest would otherwise clip. `digest is None` keeps the
        original episode path bit-for-bit (hermes on_session_end unchanged).
        """
        if digest is not None:
            body = digest.strip()
            if not body:
                return None
        else:
            episodes = self.get_recent_episodes(limit=max_episodes, session_id=session_id)
            if not episodes or len(episodes) < min_episodes:
                return None
            body = "\n".join(
                f"{e['role'].upper()}: {e['content']}" for e in episodes
            )
        if structured:
            base_prompt = structured_prompt or _DEFAULT_STRUCTURED_PROMPT
            final_prompt = f"{base_prompt}\n\nSESSION LOG:\n{body}\n\nJSON:"
        else:
            base_prompt = prompt or (
                "Summarise the session below as ONE short paragraph of durable "
                "autobiographical memory - what the user and assistant worked on and "
                "decided together, the kind of thing worth remembering next session. "
                "Frame it as a remembered summary, not a transcript; keep only the "
                "throughline, drop turn-by-turn detail; never invent anything not in "
                "the log. Output ONLY the paragraph, no preamble."
            )
            final_prompt = f"{base_prompt}\n\nSESSION LOG:\n{body}\n\nSUMMARY:"
        try:
            payload = {"model": reason_model, "prompt": final_prompt,
                       "stream": False, "options": {"temperature": 0.3}}
            req = urllib.request.Request(
                f"{ollama_endpoint}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with reason_slot(), urllib.request.urlopen(req, timeout=300.0) as response:
                raw = json.loads(response.read().decode("utf-8")).get("response", "")
        except Exception as e:
            logger.debug("Session summarisation LLM call failed (non-fatal): %s", e)
            return None
        # Reuse the shared cleaner to strip <think> blocks / code fences.
        cleaned = self._clean_llm_json(raw).strip()
        if structured:
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and str(parsed.get("throughline", "")).strip():
                th = str(parsed.get("throughline", "")).strip()
                dec = parsed.get("decisions") or []
                opn = parsed.get("open_loops") or []
                clo = parsed.get("closed") or []
                top = parsed.get("topics") or []
                blob = _render_narrative_blob(th, dec, opn, clo)
                return self.add_session_summary(
                    session_id, blob, started_cycle=started_cycle,
                    ended_cycle=ended_cycle, created_cycle=created_cycle, keep=keep,
                    throughline=th, decisions=dec, open_loops=opn, closed=clo, topics=top,
                )
            # JSON parse failed / no throughline: fall back to storing the raw prose.
            logger.debug("Structured narrative parse failed; storing freeform summary")
        summary = _ascii_sanitize(cleaned)
        if not summary:
            return None
        return self.add_session_summary(
            session_id, summary, started_cycle=started_cycle,
            ended_cycle=ended_cycle, created_cycle=created_cycle, keep=keep,
        )
