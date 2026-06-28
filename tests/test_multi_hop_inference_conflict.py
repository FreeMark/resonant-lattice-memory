r"""test_multi_hop_inference_conflict.py  (corrected)

Multi-hop transitive inference over a business relation graph + a CORRECT
no-write invariant + the conflict machinery.

Key fixes vs the original:
  * Uses deterministic relation phrasings ("X is located in Y") so the
    deterministic triple extractor reliably builds the chain - inference no
    longer depends on the LLM emitting the right triples.
  * The no-write invariant now counts rows BEFORE vs AFTER an actual
    infer_relations() call (the original read the count twice with nothing in
    between, so it was trivially true).
  * Conflict resolution is asserted via the real machinery (deterministic group),
    not via an HRR heuristic that does not fire on this data.
"""
import sys
import _common as C

# Deterministic chain: acme -> boston -> massachusetts -> usa
CHAIN = [
    ("Acme Corp is located in Boston", ["acme corp", "boston"]),
    ("Boston is located in Massachusetts", ["boston", "massachusetts"]),
    ("Massachusetts is located in the USA", ["massachusetts", "usa"]),
]


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    s, R, _, db = C.make_store("multi_hop.db")
    suite = C.Suite("Multi-Hop Inference + Conflict Machinery")

    for content, ents in CHAIN:
        # use_llm=False -> rely on the deterministic 'located in' extractor
        C.add_fact(s, R, content, category="business", entities=ents, session="mh",
                   with_relations=True, use_llm=False)
    # a couple of payment facts (entity-grounded, no relation needed for the chain)
    C.add_fact(s, R, "Approved payment for Acme: 4050 cents via link-cli --request-approval.",
               category="spend", entities=["acme corp"], session="mh")

    rels = s.get_relations(subject="acme corp")
    suite.report("acme relations", [(r["subject"], r["relation"], r["object"]) for r in rels])
    suite.hard("deterministic triple (acme corp, located_in, boston) extracted",
               any(r["relation"] == "located_in" and r["object"] == "boston" for r in rels))

    # ---- multi-hop inference ----
    inf = s.infer_relations(subject="acme corp", max_hops=3, max_results=10)
    objs = {x["object"]: x["hops"] for x in inf}
    suite.report("inferred from acme corp", objs)
    suite.hard("inference reaches boston (1 hop is direct; >=2 via chain)",
               any(x["object"] == "boston" for x in rels) or "boston" in objs)
    suite.hard("transitive inference reaches massachusetts (2 hops)",
               objs.get("massachusetts") == 2, f"hops={objs.get('massachusetts')}")
    suite.hard("transitive inference reaches usa (3 hops)",
               objs.get("usa") == 3, f"hops={objs.get('usa')}")

    # ---- no-write invariant (measured around an ACTUAL infer call) ----
    fr0 = s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    sf0 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    _ = s.infer_relations(subject="acme corp", max_hops=3, max_results=10)
    fr1 = s._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    sf1 = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    suite.hard("inference wrote NOTHING (no-write invariant)",
               fr0 == fr1 and sf0 == sf1,
               f"fact_relations {fr0}->{fr1}, semantic_facts {sf0}->{sf1}")

    # ---- conflict machinery (deterministic) ----
    # Divergent phrasings so the two rows don't merge on insert (>=0.95 cosine).
    aw, w = C.add_fact(s, R, "Sales records the Acme enterprise deal at 405000 cents per month.",
                       entities=["acme corp"], session="mh")
    al, l = C.add_fact(s, R, "A billing-system discrepancy instead shows the Acme monthly figure as 450000 cents.",
                       entities=["acme corp"], session="mh")
    suite.hard("two distinct conflicting deal facts exist (both added as separate rows)",
               aw == "added" and al == "added" and w != l, f"aw={aw} w={w} | al={al} l={l}")
    C.make_conflict_group(s, w, l, group="cg-deal", since_cycle=1, now_cycle=4)
    pend = s.get_pending_conflicts(min_age_cycles=0)
    suite.hard("disputed deal surfaces as pending conflict",
               len(pend) == 1 and pend[0]["conflict_group_id"] == "cg-deal", f"{len(pend)} group(s)")
    res = s.resolve_conflict(w, current_cycle=4)
    suite.hard("conflict resolved to the correct (405000) fact + loser superseded",
               res and res.get("winner_id") == w and l in res.get("superseded", []), res)

    s.close()
    return suite.finish("multi_hop_results.md")


if __name__ == "__main__":
    sys.exit(main())
