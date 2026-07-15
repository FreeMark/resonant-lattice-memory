#!/usr/bin/env python3
"""Inspect ONE fact in the grok agent's OWN lattice, by id (read-only).

Combines the two id-native reads the conflict/curation board wants:
  - get_fact(id)          : the full stored row (content, category, tier, resonance,
                            pinned, source, conflict_group_id, learned/confirmed cycles)
  - get_fact_history(id)  : its supersession lineage -- superseded_by_chain (the path
                            toward current belief) and replaced (the predecessors it retired)

So 'show me #227 in full' and 'what superseded what, when' come back in one call. Prints a
JSON result line. Touches nothing.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--session", default="grok-inspect")
    args = ap.parse_args()
    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        fact = store.get_fact(args.id)
        if not fact:
            print(json.dumps({"ok": False, "error": f"no fact with id {args.id}"}))
            return
        history = store.get_fact_history(args.id) or {}
        print(json.dumps({
            "ok": True,
            "fact": fact,
            "superseded_by_chain": history.get("superseded_by_chain", []),
            "replaced": history.get("replaced", []),
        }, default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
