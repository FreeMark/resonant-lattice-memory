#!/usr/bin/env python3
"""Relational recall over the grok agent's OWN triple graph (read-only, Phase 5b).

Answers "how is X connected / what does X do / what runs on X" from the (subject, relation,
object) graph the operational relation vocabulary builds. Pass a free-text --query, or any of
--subject/--relation/--object slots. Returns graph matches (exact) and HRR fuzzy near-matches,
each with the source fact. Complementary to rlm_search (semantic) and rlm_entity (co-occurrence).
Prints a JSON result line."""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=None)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--relation", default=None)
    ap.add_argument("--object", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--session", default="grok-relational")
    args = ap.parse_args()
    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        if not any([args.query, args.subject, args.relation, args.object]):
            q = sys.stdin.buffer.read().decode("utf-8", "replace").strip()
            args.query = q or None
        results = store.relational_recall(
            subject=args.subject, relation=args.relation, object=args.object,
            query=args.query, max_results=args.k,
            hrr_floor=float(getattr(prov, "_relation_recall_hrr_floor", 0.4) or 0.4),
            aliases=getattr(prov, "_entity_aliases", None))
        print(json.dumps({"ok": True, "count": len(results), "results": results}, default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
