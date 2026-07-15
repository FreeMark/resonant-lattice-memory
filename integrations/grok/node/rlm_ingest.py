#!/usr/bin/env python3
"""Ingest a grok `chat_history.jsonl` into the RLM lattice - standalone, no hermes gateway.

Flow (mirrors an overnight trainer, per-window):
  parse chat_history.jsonl -> keep user/assistant turns (system/reasoning/tool_result gated out)
  -> chunk into windows of ~reflection_frequency*2 turns (what one epoch mines)
  -> for each window: add_episode(...) under a window session_id, then _run_consolidation_epoch
  -> report per-window and total `born` facts.

Usage: rlm_ingest.py <chat_history.jsonl> [--session-id ID] [--max-windows N] [--dry-run]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider, fact_count  # noqa: E402

INCLUDE = {"user": "user", "assistant": "assistant"}  # skip system / reasoning / tool_result


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


def parse_turns(path):
    turns = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            t = o.get("type") or o.get("role")
            if t not in INCLUDE:
                continue
            text = coerce_text(o.get("content")).strip()
            if text:
                turns.append((INCLUDE[t], text))
    return turns


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat_history")
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--max-windows", type=int, default=0, help="cap windows (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = os.path.abspath(args.chat_history)
    base_sid = args.session_id or os.path.basename(os.path.dirname(path)) or "grok-ingest"
    turns = parse_turns(path)
    print(f"[ingest] file={path}")
    print(f"[ingest] base_session={base_sid}  usable turns={len(turns)} "
          f"(user/assistant only; system/reasoning/tool_result gated out)")

    if args.dry_run:
        for r, t in turns[:8]:
            print(f"   {r:9} {t[:100]!r}")
        return

    prov = build_provider(session_id=base_sid)
    store = prov._store
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

    # Rolling narrative pass: ONE autobiographical paragraph over this whole ingest, so the agent
    # wakes up with continuity. summarize_session is session-scoped and caps at ~40 episodes, so we
    # stage the transcript under a throwaway narrative session, summarize it (narrative_model /
    # narrative_endpoint if set, else the relation model), then delete those episodes. It runs AFTER
    # the dream, so the staged episodes never trigger consolidation-debt. Non-fatal.
    if turns and getattr(prov, "_enable_narrative", False):
        nmodel = prov._config.get("narrative_model") or prov._relation_model
        nep = prov._config.get("narrative_endpoint") or prov._ollama_endpoint_relation
        nsid = f"{base_sid}-narrative"
        ns = time.time()
        try:
            for role, text in turns:
                store.add_episode(nsid, role, text)
            sid = store.summarize_session(
                nmodel, nep, nsid,
                prompt=getattr(prov, "_narrative_prompt", None),
                ended_cycle=prov._memory_cycle, created_cycle=prov._memory_cycle,
                keep=getattr(prov, "_narrative_keep", 40),
                min_episodes=getattr(prov, "_narrative_min_episodes", 2))
            store._conn.execute("DELETE FROM episodes WHERE session_id = ?", (nsid,))
            store._conn.commit()
            tag = f"written #{sid}" if sid else "skipped (too thin)"
            print(f"[ingest] narrative {tag} via {nmodel} ({time.time()-ns:.0f}s)")
        except Exception as e:
            print(f"[ingest] narrative FAILED (non-fatal): {e}")

    after_total = fact_count(store)
    print(f"\n[ingest] === DONE facts {before_total} -> {after_total}  "
          f"born={after_total-before_total}  ({time.time()-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
