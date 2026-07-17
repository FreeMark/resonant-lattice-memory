#!/usr/bin/env python3
"""Read-only: return the agent's OWN recent session narratives - the P1 structured arc
(throughline / decisions / open_loops / closed / topics + created_cycle + historical),
newest first. Prints one JSON line. build_provider opens the store (running migrations),
so the structured columns are guaranteed present."""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    try:
        p = build_provider(session_id="narrative")
        rows = p._store.get_recent_narrative(limit=max(1, args.limit), chronological=False)
        out = [{
            "summary_id": r.get("summary_id"),
            "created_cycle": r.get("created_cycle"),
            "historical": r.get("historical", 0),
            "throughline": r.get("throughline"),
            "decisions": r.get("decisions") or [],
            "open_loops": r.get("open_loops") or [],
            "closed": r.get("closed") or [],
            "topics": r.get("topics") or [],
            "summary": r.get("summary"),
        } for r in rows]
        print(json.dumps({"ok": True, "count": len(out), "narratives": out}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
