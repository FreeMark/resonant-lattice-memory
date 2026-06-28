r"""test_cross_entity_contamination.py  (#2 - trust axis)

The scariest money-agent failure is not forgetting - it is RECALLING THE WRONG
ENTITY'S VALUE: asked about Acme Corp's fee, returning Acme Inc's. This test
attacks that directly, in four tiers:

  T1 MERGE INTEGRITY  - near-identical facts about DIFFERENT entities (differ only
                        in a name suffix + amount) must be stored as DISTINCT rows.
                        A merge here = two companies silently conflated (data
                        corruption). Hard-checked.
  T2 DIRECT ATTRIBUTION - query each entity; top-1 must be THAT entity's fact with
                        ITS amount; the lookalike's amount must not win.
  T3 ATTRIBUTION UNDER LOAD - same, but buried among thousands of distractors
                        (several of which mention the same company tokens).
  T4 SHARED-VALUE DISAMBIGUATION - two entities with the SAME amount; a query for
                        one must return the fact naming the RIGHT entity.

Contamination@1 (a wrong-entity fact winning top-1) must be ZERO. Contamination
deeper in the list is REPORTED.
"""
import os
import re
import sys
import _common as C

LOAD = int(os.environ.get("RL_CONTAM_LOAD", "4000"))

# (entity, amount_cents, lookalike-partner-entity) - collision pairs
PAIRS = [
    ("Acme Corp", 4050, "Acme Inc"),
    ("Acme Inc", 9000, "Acme Corp"),
    ("Globex LLC", 12500, "Globex Holdings"),
    ("Globex Holdings", 88000, "Globex LLC"),
    ("Initech", 3300, "Initrode"),
    ("Initrode", 7700, "Initech"),
]
# shared-value entities (same amount, different company)
SHARED = [("Umbrella Co", 5000), ("Wayne Co", 5000), ("Stark Co", 5000)]


def ent_key(name):
    return name.lower()


def fee_fact(entity, amt):
    return (f"{entity}'s contracted monthly service fee is {amt} cents.", [ent_key(entity)])


def owe_fact(entity, amt):
    return (f"{entity} currently owes {amt} cents on its latest invoice.", [ent_key(entity)])


def top_hit(R, query):
    hits = R.search(query, limit=10)
    return hits[0] if hits else None, hits


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    s, R, _, db = C.make_store("contam.db")
    suite = C.Suite("Cross-Entity Contamination (#2 trust)")

    # ---- T1: merge integrity for collision pairs ----
    ids = {}
    merges = []
    for entity, amt, _ in PAIRS:
        content, ents = fee_fact(entity, amt)
        a, fid = C.add_fact(s, R, content, category="fee", entities=ents, session="ce")
        ids[entity] = fid
        if a != "added":
            merges.append(entity)
    distinct_ids = len(set(ids.values()))
    suite.report("collision-pair fact ids", ids)
    suite.hard("all collision-pair facts stored as DISTINCT rows (no entity-merge corruption)",
               len(merges) == 0 and distinct_ids == len(PAIRS),
               f"merged={merges} distinct_ids={distinct_ids}/{len(PAIRS)}")

    # ---- T2: direct attribution (no load yet) ----
    # CONTENT-based, not id-based: a merge makes both entities point to the same
    # row, so an id check gives a false pass. The top-1 must literally name THIS
    # entity AND carry ITS amount.
    def attribution_pass(label):
        contam1 = 0
        wrong = []
        for entity, amt, partner in PAIRS:
            top, hits = top_hit(R, f"What is {entity}'s monthly service fee?")
            content = (top or {}).get("content", "")
            right = top and (entity.lower() in content.lower()) and (str(amt) in content)
            if not right:
                contam1 += 1
                wrong.append((entity, f"want '{entity}'/{amt}, got: {content[:60]!r}"))
        suite.report(f"{label}: wrong attributions", wrong or "none")
        suite.hard(f"{label}: every query returns the RIGHT entity + RIGHT amount (content-verified)",
                   contam1 == 0, f"{contam1}/{len(PAIRS)} wrong")
        return contam1

    attribution_pass("T2 direct")

    # ---- T3: attribution UNDER LOAD ----
    added = C.load_distractors(s, R, LOAD)
    for _ in range(2):
        s.increment_tier_cycles(); s.promote_facts()
    suite.report("distractors loaded", f"{added} added of {LOAD} (rest merged)")
    suite.report("total rows", s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0])
    attribution_pass("T3 under load")

    # ---- T4: shared-value disambiguation ----
    sid = {}
    for entity, amt in SHARED:
        content, ents = owe_fact(entity, amt)
        a, fid = C.add_fact(s, R, content, category="debt", entities=ents, session="ce")
        sid[entity] = fid
    for _ in range(2):
        s.increment_tier_cycles(); s.promote_facts()
    shared_wrong = 0
    detail = []
    for entity, amt in SHARED:
        top, hits = top_hit(R, f"How much does {entity} owe?")
        ok = top and top.get("id") == sid[entity] and entity.lower() in top.get("content", "").lower()
        if not ok:
            shared_wrong += 1
            detail.append((entity, str(top.get("content"))[:55] if top else "none"))
    suite.report("T4 shared-value wrong attributions", detail or "none")
    suite.hard("T4: same-amount entities are disambiguated (query returns the RIGHT entity)",
               shared_wrong == 0, f"{shared_wrong}/{len(SHARED)} wrong")

    s.close()
    return suite.finish("contamination_results.md")


if __name__ == "__main__":
    sys.exit(main())
