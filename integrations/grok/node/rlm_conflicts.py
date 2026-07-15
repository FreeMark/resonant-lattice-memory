#!/usr/bin/env python3
"""Conflict management for the grok agent's OWN lattice.

  --op list     : list pending (unresolved) conflict groups
  --op resolve  : pick a winner (--winner-id); the other members retire as superseded history
  --op dismiss  : mark a group a false positive (--group-id); all members kept and unlocked

Prints a JSON result. Writes only to the agent's own lattice.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["list", "resolve", "dismiss"], required=True)
    ap.add_argument("--winner-id", type=int, default=0)
    ap.add_argument("--group-id", default=None)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    try:
        p = build_provider(session_id="conflicts")
        s = p._store
        if args.op == "list":
            groups = s.get_pending_conflicts(min_age_cycles=0, limit=args.limit)
            print(json.dumps({"ok": True, "count": len(groups), "conflicts": groups}, default=str))
        elif args.op == "resolve":
            if not args.winner_id:
                print(json.dumps({"ok": False, "error": "resolve needs --winner-id"}))
                return
            res = s.resolve_conflict(args.winner_id, current_cycle=p._memory_cycle)
            print(json.dumps({"ok": res is not None, "result": res}, default=str))
        else:  # dismiss
            if not args.group_id:
                print(json.dumps({"ok": False, "error": "dismiss needs --group-id"}))
                return
            res = s.dismiss_conflict(group_id=args.group_id, current_cycle=p._memory_cycle)
            print(json.dumps({"ok": res is not None, "result": res}, default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
