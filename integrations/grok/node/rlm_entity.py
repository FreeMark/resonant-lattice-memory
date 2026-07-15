#!/usr/bin/env python3
"""Walk the grok agent's OWN entity graph (read-only). Two modes:

  (entity mode, default): the entity NAME is read from STDIN (utf-8). Returns facts linked to
                    that entity (ranked by resonance) PLUS the entities that co-occur with it
                    (get_facts_for_entity + get_related_entities). 'everything about
                    nomic-embed / the node / Grok@Rivernest', with neighbours.
  --fact-id N     : the entities linked to one fact (get_entities_for_fact).

Complementary to rlm_search (semantic) -- this is graph traversal, not similarity. Prints a
JSON result line. Touches nothing.
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlm_common import build_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-id", type=int, default=0)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min-shared", type=int, default=2)
    ap.add_argument("--session", default="grok-entity")
    args = ap.parse_args()

    entity = "" if args.fact_id else sys.stdin.buffer.read().decode("utf-8", "replace").strip()
    try:
        prov = build_provider(session_id=args.session)
        store = prov._store
        if args.fact_id:
            entities = store.get_entities_for_fact(args.fact_id)
            print(json.dumps({"ok": True, "mode": "fact", "fact_id": args.fact_id,
                              "entities": entities}, default=str))
        elif entity:
            facts = store.get_facts_for_entity(entity, limit=args.k)
            related = store.get_related_entities(entity, min_shared=args.min_shared,
                                                 limit=args.k)
            print(json.dumps({"ok": True, "mode": "entity", "entity": entity,
                              "facts": facts, "related": related}, default=str))
        else:
            print(json.dumps({"ok": False, "error": "provide an entity (stdin) or --fact-id N"}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))


if __name__ == "__main__":
    main()
