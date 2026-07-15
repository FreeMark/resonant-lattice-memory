#!/usr/bin/env python3
"""Bounded multi-hop inference over the grok agent's OWN triple graph (read-only, Phase 5c).

Chains stored triples forward from --subject (a->b->c...) up to --max-hops and returns the
DERIVED connections, each with the full supporting path of REAL stored triples and a confidence
that DECAYS per hop. Inferences are NEVER stored and are always weaker than a stored fact -- they
are surfaced as hypotheses ('the node serves granite, granite runs the narrative, so the narrative
depends on the node'), for the agent to interpret, not gospel. Optional --object filters to chains
ending there. Prints a JSON result line."""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    ap.add_argument("--object", default=None)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--session", default="grok-infer")
    args = ap.parse_args()
    subject = args.subject or sys.stdin.buffer.read().decode("utf-8", "replace").strip()
    if not subject:
        print(json.dumps({"ok": False, "error": "provide a --subject (or on stdin) to chain from"}))
        return
    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        results = store.infer_relations(subject=subject, object=args.object,
                                        max_hops=max(2, int(args.max_hops)), max_results=args.k,
                                        aliases=getattr(prov, "_entity_aliases", None))
        print(json.dumps({"ok": True, "count": len(results), "inferences": results}, default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
