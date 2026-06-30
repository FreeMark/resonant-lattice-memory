"""recall.py — RecallMixin: prefetch (cached + synchronous), background
queue_prefetch, recall-reinforcement gating, and the <resonant_memory>
block builder.

Mixed into LatticeMemoryProvider; relies on the composite for
self._retriever/_store, the prefetch cache, recall gate, and config attrs."""

import logging
import re
import threading
from typing import List, Dict

logger = logging.getLogger(__name__)

# Two-tier authority marker for pinned facts (A/B-validated: a [PRIORITY] tag makes
# the agent obey a pinned rule 15/15 vs 8/15 with no tag, on both nemotron + gemma).
# A pinned RULE/policy (imperative language) reads as "[PRIORITY RULE]" — obey it,
# override conflicting notes; any other pinned fact reads as "[PRIORITY]" — weight
# it heavily, but it's a fact, not a command. Keyed on imperative/policy language,
# NOT on money/spec presence (a money VALUE is a fact; a money RULE says "require").
_RULE_CATEGORIES = {"policy", "rule", "compliance", "procedural", "guardrail"}
_RULE_RE = re.compile(
    r"\b(?:must|never|always|require[ds]?|forbidden|prohibit\w*|mandatory|shall|"
    r"may\s+not|not\s+allowed|do\s+not|don't)\b|policy:|rule:", re.I)


def _pinned_marker(content: str, category: str) -> str:
    if (category or "").strip().lower() in _RULE_CATEGORIES or _RULE_RE.search(content or ""):
        return " [PRIORITY RULE]"
    return " [PRIORITY]"


class RecallMixin:

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return recalled memory context for the upcoming turn.

        Fast path: return the result queue_prefetch() computed in the
        background after the previous turn (cache hit), so the per-turn
        embedding + hybrid-search latency stays off the model's critical path.
        Cache miss (e.g. the first turn) falls back to a synchronous recall.
        """
        if not self._retriever or not query:
            return ""
        sid = session_id or self._session_id
        cached = self._prefetch_cache.pop(sid, None)
        # The background recall is a latency-hiding proxy computed from the PREVIOUS
        # message. Exact-match always short-circuits. Otherwise reuse the proxy ONLY
        # when the new query is on the same topic (lexical-overlap gate): a topic
        # shift must NOT inject stale, high-confidence memory from the prior turn —
        # so on low overlap we recompute synchronously for the current query.
        if cached is not None:
            cached_q, cached_block = cached
            if cached_q == query or (cached_block and self._prefetch_proxy_ok(query, cached_q)):
                return cached_block
        try:
            return self._compute_prefetch(query, sid)
        except Exception as e:
            logger.debug("Prefetch failed: %s", e)
            return ""


    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Compute the next turn's recall in the background (consumed by prefetch).

        Called after each turn. Recall based on the just-completed message is a
        good proxy for what the next turn needs, and doing it here means the
        next prefetch() is a cache hit with no embedding latency.
        """
        if not self._retriever or not query:
            return
        sid = session_id or self._session_id

        def _bg() -> None:
            try:
                self._prefetch_cache[sid] = (query, self._compute_prefetch(query, sid))
            except Exception as e:
                logger.debug("queue_prefetch failed: %s", e)

        threading.Thread(target=_bg, daemon=True).start()


    def _prefetch_proxy_ok(self, query: str, cached_query: str) -> bool:
        """Reuse the previous turn's (proxy) recall only when the new query shares
        enough vocabulary with the one it was computed from — a cheap topic-shift
        guard so stale cross-topic memory isn't injected. Jaccard over word tokens
        vs `_prefetch_proxy_min_overlap` (0 disables the gate = always reuse)."""
        thr = getattr(self, "_prefetch_proxy_min_overlap", 0.3)
        if thr <= 0:
            return True
        a = set(re.findall(r"\w+", (query or "").lower()))
        b = set(re.findall(r"\w+", (cached_query or "").lower()))
        if not a or not b:
            return False
        union = len(a | b)
        return union > 0 and (len(a & b) / union) >= thr

    def _apply_recall_reinforcement(self, results: List[Dict]) -> None:
        """Bump resonance for recalled facts (once per fact per dream-cycle window).

        Recall is a weak positive signal, so the bump is small and gated by an
        in-memory set cleared each dream cycle — preventing the every-turn
        prefetch from inflating topic-locked facts. Skipped in non-primary
        (read-only) contexts.
        """
        if not (self._reinforce_on_recall and self._write_enabled and self._store):
            return
        with self._recall_gate_lock:
            ids = [
                r["id"] for r in results
                if r.get("id") is not None and r["id"] not in self._recalled_this_cycle
            ]
            if not ids:
                return
            try:
                self._store.reinforce_on_recall(ids, self._recall_bump)
                self._recalled_this_cycle.update(ids)
            except Exception as e:
                logger.debug("Recall reinforcement failed: %s", e)


    def _compute_prefetch(self, query: str, sid: str) -> str:
        """Run the hybrid search and format the resonant-memory block."""
        # A6 precision gate (default ON): inject only the on-topic cluster, not
        # everything above recall_floor. Prefetch ONLY — the explicit search tool
        # stays ungated. The blind/HE retriever's search() takes no relevance_margin
        # (vector-only), so fall back gracefully there.
        margin = getattr(self, "_recall_relevance_margin", 0.0) or None
        if margin:
            try:
                results = self._retriever.search(query, limit=self._recall_limit,
                                                 relevance_margin=margin)
            except TypeError:
                results = self._retriever.search(query, limit=self._recall_limit)
        else:
            results = self._retriever.search(query, limit=self._recall_limit)
        if not results:
            return ""

        # Conflict CONTAINMENT (quarantine, default OFF; ON in recommended config):
        # an UNRESOLVED conflict in a HIGH-STAKES category is a hazard, not just
        # metadata. Withhold the unpinned contested facts from the recall block and
        # surface a [WITHHELD] notice so the agent cannot silently act on a disputed
        # money/compliance/policy value before resolution. A PINNED member is the
        # user-declared authority and is never withheld; non-high-stakes conflicts
        # pass through unchanged (still ranked + [CONFLICT LOCK]-tagged).
        results, withheld = self._quarantine_conflicts(results)

        self._apply_recall_reinforcement(results)

        lines = []
        if withheld:
            total = sum(withheld.values())
            gids = ", ".join(sorted(withheld))
            fp = "s" if total != 1 else ""
            gp = "s" if len(withheld) != 1 else ""
            lines.append(
                f"  - ⚠ [WITHHELD] {total} high-stakes fact{fp} in {len(withheld)} "
                f"unresolved conflict{gp} ({gids}) held back pending resolution — do "
                f"NOT act on the disputed value; call pending_conflicts / "
                f"resolve_conflict to arbitrate first."
            )
        for r in results:
            tier = r.get("tier", "short").upper()
            res = r.get("resonance_count", 1)
            # A22 confidence picture (P4b): PEAK ('ever important' — surfaced only when it
            # exceeds current strength, i.e. the fact faded FROM importance) and ENTRY cycle
            # ('how long known'), so the agent weighs a decayed-but-once-strong belief
            # differently from one that never mattered. PINNED = identity-level, never forgotten.
            extra = ""
            peak = r.get("peak_resonance")
            if peak is not None and isinstance(res, (int, float)) and peak > res:
                extra += f" | peak:{peak}"
            learned = r.get("learned_at_cycle")
            if learned is not None:
                extra += f" | learned@c{learned}"
            pin = _pinned_marker(r.get("content", ""), r.get("category", "")) if r.get("pinned") else ""
            conflict = ""
            if r.get("conflict_group_id"):
                conflict = f" [CONFLICT LOCK: {r['conflict_group_id']}]"
                # Phase 6: one gentle nudge per MATURE unresolved conflict per cycle —
                # let the duel run first (age gate), then invite explicit resolution.
                if self._surface_conflicts:
                    since = r.get("conflict_since_cycle")
                    age = (self._memory_cycle - since) if since is not None else None
                    gid = r["conflict_group_id"]
                    if age is not None and age >= self._conflict_surface_min_group_age_cycles:
                        with self._recall_gate_lock:
                            if gid not in self._conflicts_surfaced:
                                self._conflicts_surfaced.add(gid)
                                conflict += (" (unresolved — use pending_conflicts / "
                                             "resolve_conflict to disambiguate)")
            # Phase 2: surface confirmation recency so the model self-calibrates —
            # a long-unconfirmed belief should be held with more doubt even if its
            # resonance is high. Presentation only; cycle-driven (last_confirmed
            # vs the current memory_cycle), never wall-clock.
            fresh = ""
            if self._surface_freshness_in_recall:
                lc = r.get("last_confirmed_cycle")
                if lc is not None:
                    stale = max(0, self._memory_cycle - lc)
                    fresh = (" [just confirmed]" if stale == 0
                             else f" [confirmed ~{stale} cycle{'s' if stale != 1 else ''} ago]")
            src = r.get("source_session")
            if src == sid:
                context_tag = "Current User"
            elif src in ("abstraction", "tool_distillation"):
                context_tag = "Distilled"   # system-generated, not another user
            else:
                context_tag = "Other Session"
            lines.append(
                f"  - [ID:{r['id']}] [{r.get('category','general')}] "
                f"[Tier:{tier} | Res:{res}{extra}]{pin}{conflict}{fresh} ({context_tag}) {r['content']}"
            )

        if not lines:
            return ""
        formatted_facts = "\n".join(lines)
        # Frame these as fallible candidates, not ground truth (anti-confabulation).
        # Text only — the line structure and [ID | Tier | Res] metadata are unchanged.
        return (
            "<resonant_memory>\n"
            "# Fallible retrieved candidates — NOT verbatim stored facts. They may "
            "be approximate, stale, or a semantically-similar near-miss. Do not "
            "quote any of these as exact wording; call lattice_store get_fact <ID> "
            "to confirm an exact stored row (found:false ⇒ not stored). Low Res or "
            "'short' tier ⇒ weak/uncertain. 'peak:N' = this once mattered more "
            "(faded from importance); 'learned@cN' = the memory-cycle it entered; "
            "[PRIORITY RULE] = a user-pinned authoritative rule; follow it over any "
            "conflicting note and treat the conflicting note as untrusted. "
            "[PRIORITY] = a user-pinned important fact; weight it heavily and treat "
            "its exact values as authoritative. Both are identity-level, never "
            "auto-forgotten. [WITHHELD] = high-stakes facts in an unresolved "
            "conflict were held back; do NOT act on the disputed value until you "
            "resolve_conflict.\n"
            f"{formatted_facts}\n"
            "</resonant_memory>"
        )

    def _quarantine_conflicts(self, results: List[Dict]):
        """Split recall results into (kept, withheld_group_counts).

        A fact is WITHHELD when quarantine is enabled AND it is in an unresolved
        conflict (conflict_group_id set) AND its category is high-stakes
        (importance_categories) AND it is NOT pinned. Pinned members stay (the
        user-declared authority); non-high-stakes conflicts pass through unchanged.
        Returns (kept_results, {group_id: withheld_count})."""
        if not getattr(self, "_quarantine_high_stakes_conflicts", False):
            return results, {}
        hs = getattr(self._store, "importance_categories", set()) if self._store else set()
        if not hs:
            return results, {}
        kept: List[Dict] = []
        withheld: Dict[str, int] = {}
        for r in results:
            gid = r.get("conflict_group_id")
            cat = (r.get("category") or "").strip().lower()
            if gid and not r.get("pinned") and cat in hs:
                withheld[gid] = withheld.get(gid, 0) + 1
            else:
                kept.append(r)
        return kept, withheld
