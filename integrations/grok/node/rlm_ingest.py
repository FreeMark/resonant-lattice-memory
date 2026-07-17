#!/usr/bin/env python3
"""Ingest a grok `chat_history.jsonl` into the RLM lattice - standalone, no hermes gateway.

Flow (mirrors an overnight trainer, per-window):
  parse chat_history.jsonl -> keep user/assistant turns (system/reasoning/tool_result gated out)
  -> chunk into windows of ~reflection_frequency*2 turns (what one epoch mines)
  -> for each window: add_episode(...) under a window session_id, then _run_consolidation_epoch
  -> report per-window and total `born` facts.

Usage: rlm_ingest.py <chat_history.jsonl> [--session-id ID] [--max-windows N] [--dry-run]
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider, fact_count  # noqa: E402

INCLUDE = {"user": "user", "assistant": "assistant"}  # legacy flat format: skip system/reasoning/tool
# Current grok transcripts are an ACP (Agent Client Protocol) session-update stream: the role is in
# params.update.sessionUpdate and the text in params.update.content. Everything else
# (agent_thought_chunk, tool_call, tool_call_update, plan, turn_completed, ...) is skipped.
ACP_ROLE = {"user_message_chunk": "user", "agent_message_chunk": "assistant"}


def coerce_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or "")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or json.dumps(content)[:2000]
    return str(content)


def parse_turns(path, skip_lines=0):
    """Parse (role, text) turns from the transcript, skipping the first `skip_lines` raw
    lines. Returns (turns, total_lines). grok's chat_history.jsonl is append-only, so the
    first N lines of any later snapshot are byte-identical to what an earlier compact
    already ingested -- a line high-water mark lets a re-compact ingest only the new tail
    instead of re-mining the whole session."""
    raw = []
    total = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            total = i + 1
            if i < skip_lines:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            # ACP session/update envelope: role in params.update.sessionUpdate, text in .content
            params = o.get("params")
            upd = params.get("update") if isinstance(params, dict) else None
            if isinstance(upd, dict) and "sessionUpdate" in upd:
                role = ACP_ROLE.get(upd.get("sessionUpdate"))
                if not role:
                    continue
                text = coerce_text(upd.get("content")).strip()
                if text:
                    raw.append((role, text))
                continue
            # legacy flat transcript: top-level type/role + content
            t = o.get("type") or o.get("role")
            if t in INCLUDE:
                text = coerce_text(o.get("content")).strip()
                if text:
                    raw.append((INCLUDE[t], text))
    # merge consecutive same-role chunks (streaming) into one logical turn
    turns = []
    for role, text in raw:
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + text)
        else:
            turns.append((role, text))
    return turns, total


def make_windows(turns, max_turns, max_chars=12000):
    windows, cur, cur_chars = [], [], 0
    for role, text in turns:
        if cur and (len(cur) >= max_turns or cur_chars + len(text) > max_chars):
            windows.append(cur)
            cur, cur_chars = [], 0
        cur.append((role, text))
        cur_chars += len(text)
    if cur:
        windows.append(cur)
    return windows


def _ensure_watermark_table(store):
    store._conn.execute(
        "CREATE TABLE IF NOT EXISTS ingest_watermarks ("
        "session_key TEXT PRIMARY KEY, lines_ingested INTEGER NOT NULL, "
        "turns_ingested INTEGER, updated_at TEXT)")
    store._conn.commit()


def _get_watermark(store, key):
    row = store._conn.execute(
        "SELECT lines_ingested FROM ingest_watermarks WHERE session_key = ?", (key,)).fetchone()
    return int(row[0]) if row else 0


def _set_watermark(store, key, lines, turns):
    store._conn.execute(
        "INSERT OR REPLACE INTO ingest_watermarks "
        "(session_key, lines_ingested, turns_ingested, updated_at) "
        "VALUES (?, ?, ?, datetime('now'))", (key, lines, turns))
    store._conn.commit()


def _session_key(base_sid):
    # ingest session-id is '<grok-session-uuid>_<HHMMSS>'; key the watermark on the stable
    # grok session so repeated compacts of the same session share one high-water mark.
    return re.sub(r"_\d{6}$", "", base_sid)


# Categories that mark durable / resumable material - surfaced first in the narrative digest.
_NARRATIVE_PRIORITY_CATS = {
    "decision", "constraint", "requirement", "best_practice", "open_loop",
    "goal", "project", "rule", "preference", "plan",
}
_MAX_DIGEST_CHARS = 60000  # generous; granite runs at 72420 ctx. Guard against pathological sessions.


def _sample_spine(items, n):
    """Head/mid/tail sample of `items` down to ~n, order preserved, so the spine shows the
    session's flow (opening intent -> mid pivots -> latest) instead of only its tail."""
    if n <= 0 or len(items) <= n:
        return list(items)
    k = max(1, n // 3)
    head, tail = items[:k], items[-k:]
    mid_n = n - 2 * k
    if mid_n > 0:
        start = max(k, (len(items) - mid_n) // 2)
        mid = items[start:start + mid_n]
    else:
        mid = []
    return head + mid + tail


def _build_narrative_digest(store, base_sid, n_windows, turns,
                            per_window=10, spine_turns=12, spine_char_cap=240):
    """Build a hierarchical, born-fact-grounded digest of the WHOLE ingest for the narrative
    pass, replacing the tail-40-episodes input that clips long multi-window sessions.

    Three sections, all from material the ingest already produced (no extra LLM calls):
      1. BORN FACTS BY WINDOW - per window, its highest-signal born facts (the belief store's
         own distillation), ordered pinned -> priority-category -> high-resonance.
      2. OPEN / DECISION-LIKE - the priority-category facts rolled up across all windows.
      3. SESSION SPINE - a head/mid/tail sample of user turns, for narrative flow.
    Returns the digest string (empty when there is nothing to narrate)."""
    window_blocks, open_like, seen_open = [], [], set()
    for i in range(n_windows):
        wsid = f"{base_sid}-w{i:02d}"
        rows = store._conn.execute(
            "SELECT content, category, COALESCE(resonance_count, 0) AS rc, "
            "COALESCE(pinned, 0) AS pinned FROM semantic_facts WHERE source_session = ?",
            (wsid,)).fetchall()
        if not rows:
            continue
        rows = sorted(
            rows,
            key=lambda r: (int(r["pinned"] or 0),
                           1 if (r["category"] or "").lower() in _NARRATIVE_PRIORITY_CATS else 0,
                           float(r["rc"] or 0)),
            reverse=True)
        lines = [f"WINDOW {i:02d}:"]
        for r in rows[:per_window]:
            cat = (r["category"] or "fact").lower()
            content = " ".join((r["content"] or "").split())
            lines.append(f"  - [{cat}] {content}")
            if cat in _NARRATIVE_PRIORITY_CATS and content not in seen_open:
                seen_open.add(content)
                open_like.append(f"  - [{cat}] {content}")
        window_blocks.append("\n".join(lines))

    sections = []
    if window_blocks:
        sections.append("BORN FACTS BY WINDOW (the session's distilled beliefs, in order):\n"
                        + "\n".join(window_blocks))
    if open_like:
        sections.append("OPEN / DECISION-LIKE (rolled up across the whole session):\n"
                        + "\n".join(open_like[:24]))
    user_turns = [t for role, t in turns if role == "user"] or [t for _, t in turns]
    if user_turns:
        spine = _sample_spine(user_turns, spine_turns)
        spine_lines = ["  - " + " ".join(t.split())[:spine_char_cap] for t in spine]
        sections.append("SESSION SPINE (user intents, head/mid/tail):\n" + "\n".join(spine_lines))

    digest = "\n\n".join(sections)
    if len(digest) > _MAX_DIGEST_CHARS:
        digest = digest[:_MAX_DIGEST_CHARS] + "\n[... digest truncated ...]"
    return digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat_history")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--max-windows", type=int, default=0, help="cap windows (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ignore-watermark", action="store_true",
                    help="re-ingest the whole snapshot even if a session watermark exists")
    args = ap.parse_args()

    path = os.path.abspath(args.chat_history)
    base_sid = args.session_id or os.path.basename(os.path.dirname(path)) or "grok-ingest"
    print(f"[ingest] file={path}")

    if args.dry_run:
        turns, total = parse_turns(path)
        print(f"[ingest] base_session={base_sid}  usable turns={len(turns)}  ({total} lines)")
        for r, t in turns[:8]:
            print(f"   {r:9} {t[:100]!r}")
        return

    prov = build_provider(session_id=base_sid)
    store = prov._store
    session_key = _session_key(base_sid)
    _ensure_watermark_table(store)
    wm = 0 if args.ignore_watermark else _get_watermark(store, session_key)
    turns, total_lines = parse_turns(path, skip_lines=wm)
    print(f"[ingest] base_session={base_sid}  session_key={session_key}")
    if wm:
        print(f"[ingest] watermark={wm} lines already ingested; snapshot={total_lines} lines "
              f"-> {len(turns)} NEW usable turns (re-compact: skipping ingested prefix)")
    else:
        print(f"[ingest] usable turns={len(turns)}  ({total_lines} lines; no prior watermark; "
              f"user/assistant only, system/reasoning/tool_result gated out)")

    if not turns:
        print(f"[ingest] === nothing new since line {wm} (snapshot {total_lines} lines); "
              f"no windows to ingest ===")
        _set_watermark(store, session_key, total_lines, 0)
        return

    win_turns = max(6, int(getattr(prov, "_reflection_frequency", 5)) * 2)
    windows = make_windows(turns, win_turns)
    if args.max_windows > 0:
        windows = windows[:args.max_windows]
    print(f"[ingest] db={store.db_path}")
    print(f"[ingest] reason_model={prov._reason_model} @ {prov._ollama_endpoint_reason}")
    print(f"[ingest] window_size={win_turns} turns  windows={len(windows)}")

    before_total = fact_count(store)
    t0 = time.time()
    for i, win in enumerate(windows):
        wsid = f"{base_sid}-w{i:02d}"
        for role, text in win:
            store.add_episode(wsid, role, text)
        b0 = fact_count(store)
        ws = time.time()
        prov._run_consolidation_epoch(wsid, force_blocking=True, suppress_dream=True)
        b1 = fact_count(store)
        print(f"[ingest] window {i+1:02d}/{len(windows)} sid={wsid} "
              f"turns={len(win)} born={b1-b0} facts={b1} ({time.time()-ws:.0f}s)")
        # Phase-4: mine the RAW window transcript for conversational relations (dependency /
        # decision / usage edges the per-fact pass loses), strict-bound, attached to this
        # window's born facts. Off unless relation_extract_from_transcript + a vocabulary are set.
        if getattr(prov, "_relation_extract_from_transcript", False) and \
                getattr(prov, "_relation_vocabulary", None):
            born = store._conn.execute(
                "SELECT id, content FROM semantic_facts WHERE source_session = ?", (wsid,)).fetchall()
            if born:
                win_text = "\n".join(f"{role}: {t}" for role, t in win)
                try:
                    nrel = store.extract_transcript_relations(
                        win_text, [(r["id"], r["content"]) for r in born],
                        min_confidence=prov._relation_min_confidence,
                        reason_model=prov._relation_model,
                        ollama_endpoint=prov._ollama_endpoint_relation,
                        vocabulary=prov._relation_vocabulary,
                        examples=getattr(prov, "_relation_examples", None),
                        aliases=getattr(prov, "_entity_aliases", None),
                        llm_prompt=getattr(prov, "_relation_llm_prompt", None))
                    print(f"[ingest]   transcript-relations +{nrel}")
                except Exception as e:
                    print(f"[ingest]   transcript-relations FAILED (non-fatal): {e}")

    # Post-ingest dream pass: ONE full Hebbian maintenance cycle over everything just ingested
    # (tier-dwell, decay [unpinned only; pinned facts exempt], promotion, conflict detection,
    # prune, and any enabled abstraction / gist / self-model). The per-window epochs above use
    # suppress_dream=True (fast extract + relations); the dream runs ONCE here so the temporal
    # dynamics fire per ingest session, not per window.
    if windows:
        ds = time.time()
        try:
            prov._run_dream_cycle(wait_lock=True)
            print(f"[ingest] dream cycle {getattr(prov, '_dream_cycle_count', '?')} ran "
                  f"({time.time()-ds:.0f}s)")
        except Exception as e:
            print(f"[ingest] dream cycle FAILED (non-fatal): {e}")

    # Rolling narrative pass: ONE autobiographical paragraph over this WHOLE ingest, so the agent
    # wakes up with continuity. Feeding summarize_session the raw transcript only lets it see the
    # tail-40 episodes (granite runs with --context-shift, which clips the oldest), so a long
    # multi-window ingest would narrate only the last hinge. Instead we build a hierarchical digest
    # from what the ingest already produced - per-window born facts (the belief store's own
    # distillation) + open/decision roll-up + a head/mid/tail user-turn spine - and hand THAT to
    # summarize_session via digest=. No extra LLM calls, grounded in the attested facts, full-session
    # coverage. Uses narrative_model / narrative_endpoint if set, else the relation model. Runs AFTER
    # the dream so the born-fact set reflects the post-consolidation survivors. Non-fatal.
    if turns and getattr(prov, "_enable_narrative", False):
        nmodel = prov._config.get("narrative_model") or prov._relation_model
        nep = prov._config.get("narrative_endpoint") or prov._ollama_endpoint_relation
        structured = bool(prov._config.get("narrative_structured", True))
        nsid = f"{base_sid}-narrative"
        ns = time.time()
        try:
            digest = _build_narrative_digest(store, base_sid, len(windows), turns)
            sid = store.summarize_session(
                nmodel, nep, nsid,
                prompt=getattr(prov, "_narrative_prompt", None),
                structured=structured,
                structured_prompt=prov._config.get("narrative_structured_prompt"),
                digest=digest,
                ended_cycle=prov._memory_cycle, created_cycle=prov._memory_cycle,
                keep=getattr(prov, "_narrative_keep", 40))
            if sid:
                # P1.2 temporal framing: the new narrative is CURRENT; older ones become historical
                # so the projection renders the newest as current status, not a stale backlog voice.
                store.mark_prior_narratives_historical(keep_current=1)
            tag = f"written #{sid}" if sid else "skipped (too thin)"
            mode = "structured" if structured else "freeform"
            print(f"[ingest] narrative {tag} ({mode}) via {nmodel} "
                  f"(hierarchical digest {len(digest)} chars, {len(windows)} windows) "
                  f"({time.time()-ns:.0f}s)")
        except Exception as e:
            print(f"[ingest] narrative FAILED (non-fatal): {e}")

    after_total = fact_count(store)
    _set_watermark(store, session_key, total_lines, len(turns))
    print(f"\n[ingest] === DONE facts {before_total} -> {after_total}  "
          f"born={after_total-before_total}  ({time.time()-t0:.0f}s) "
          f"[watermark {session_key} = {total_lines} lines] ===")


if __name__ == "__main__":
    main()
