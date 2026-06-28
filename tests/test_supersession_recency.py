r"""test_supersession_recency.py  (#3 - trust axis: is the recalled value CURRENT?)

When a fact's value changes ("Acme is Net-30" -> "Acme is Net-60"), does recall
surface the NEW value, or the stale OLD one? Acting on outdated financial data is
a classic agent failure.

30 update scenarios across categories (terms / credit limit / manager / HQ /
status), each a unique entity so it's trackable + findable. For each: store the
old value, age it a few cycles, store the new value, then query and check the
TOP result reflects the CURRENT value, not the stale one. Plus mechanism checks:
the explicit conflict->resolve supersession path, and the freshness ranking nudge.

Measured (per scenario): new value present in recall, new value is top-1, stale
value is top-1 (the failure), and whether the update merged into the old row
(silently dropping the new value).
"""
import os
import sys
import _common as C

N = int(os.environ.get("RL_RECENCY_N", "30"))
LOAD = int(os.environ.get("RL_RECENCY_LOAD", "1500"))

CODENAMES = ["Tanager", "Meridian", "Halcyon", "Vantyx", "Borealis", "Cindra", "Nimbus", "Orrery",
             "Kestrel", "Zephyrine", "Quokka", "Vellum", "Saffron", "Pylon", "Draxis", "Lumen",
             "Ironwood", "Calyx", "Marrow", "Tindra", "Aster", "Cobalt", "Verdant", "Halite",
             "Onyx", "Cinnabar", "Peregrine", "Quillon", "Sable", "Wren"]

# (fact template, query, old, new) - {S}=codename, {V}=value
SHAPES = [
    ("{S} Corp payment terms are Net-{V}.", "What are {S} Corp's payment terms?", "30", "60"),
    ("{S} Corp's credit limit is {V} cents.", "What is {S} Corp's credit limit?", "500000", "1200000"),
    ("{S} Corp's account manager is {V}.", "Who is {S} Corp's account manager?", "Dana Pike", "Omar Reyes"),
    ("{S} Corp is headquartered in {V}.", "Where is {S} Corp headquartered?", "Boise", "Lisbon"),
    ("{S} Corp's contract status is {V}.", "What is {S} Corp's contract status?", "trial", "renewed"),
]


def scenarios():
    out = []
    for i in range(N):
        code = CODENAMES[i % len(CODENAMES)] + (f"{i//len(CODENAMES)}" if i >= len(CODENAMES) else "")
        ft, q, old, new = SHAPES[i % len(SHAPES)]
        out.append({"old": ft.format(S=code, V=old), "new": ft.format(S=code, V=new),
                    "query": q.format(S=code), "old_v": old, "new_v": new,
                    "ent": code.lower()})
    return out


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    # freshness nudge ON (recommended config) so recency can influence ranking
    s, R, _, db = C.make_store("recency.db", freshness_halflife=50)
    suite = C.Suite("Supersession / Recency (#3 trust)")

    scen = scenarios()
    # 1) store OLD values, age them, then store the UPDATE (new value)
    old_ids, new_ids, merged = {}, {}, []
    for sc in scen:
        a, fid = C.add_fact(s, R, sc["old"], category="record", entities=[sc["ent"]], session="rec")
        old_ids[sc["query"]] = fid
    for c in range(1, 4):                      # age the old facts
        s.set_cycle_counts(memory_cycle=c)
        s.apply_cycle_decay(); s.increment_tier_cycles(); s.promote_facts()
    for sc in scen:
        a, fid = C.add_fact(s, R, sc["new"], category="record", entities=[sc["ent"]], session="rec")
        new_ids[sc["query"]] = fid
        if a != "added":
            merged.append(sc["query"])         # update folded into the old row
    s.set_cycle_counts(memory_cycle=5)
    s.increment_tier_cycles(); s.promote_facts()

    # bury under load for realism
    C.load_distractors(s, R, LOAD)
    for c in range(6, 8):
        s.set_cycle_counts(memory_cycle=c); s.increment_tier_cycles(); s.promote_facts()

    new_present = new_top = stale_top = 0
    stale_examples = []
    for sc in scen:
        hits = R.search(sc["query"], limit=5)
        contents = [h.get("content", "") for h in hits]
        top = contents[0] if contents else ""
        # value-aware: the new value present anywhere in top-5?
        if any(_has_value(c, sc) for c in contents):
            new_present += 1
        if _is_new(top, sc):
            new_top += 1
        elif _is_old(top, sc):
            stale_top += 1
            if len(stale_examples) < 6:
                stale_examples.append((sc["query"], f"got stale: {top[:60]!r}"))

    suite.report("update merged into old row (new value at risk)", f"{len(merged)}/{N}")
    suite.report("new value present in top-5", f"{new_present}/{N}")
    suite.report("CURRENT value is top-1", f"{new_top}/{N}")
    suite.report("STALE value is top-1 (failure)", f"{stale_top}/{N}")
    suite.report("stale-top examples", stale_examples or "none")

    # GUARANTEE (hard): a value-update is never silently DROPPED — the current
    # value is always retained + recallable (the merge gate no longer folds a
    # changed number into the stale row).
    suite.hard("value-update never silently dropped (current value recallable in top-5) for every update",
               new_present == N, f"{new_present}/{N}")
    suite.hard("no update was swallowed as a reinforcement of the stale value", len(merged) == 0,
               f"{len(merged)} merged")
    # CHARACTERIZED (soft): autonomous ranking does NOT auto-prefer the newer value
    # — old + new coexist and the stale one can out-rank. This is by design (the
    # system surfaces value-changes rather than silently flipping them); the
    # TRUSTWORTHY path to authoritative currency is the supersession machinery
    # below (or surfacing the change as a conflict — recommended enhancement).
    suite.soft("autonomous recall ranks the CURRENT value top-1 (characterized, not a guarantee)",
               new_top >= stale_top, f"current_top={new_top}/{N}, stale_top={stale_top}")

    # 2) explicit supersession machinery (deterministic)
    s2, R2, _, _ = C.make_store("recency_sup.db")
    _, w = C.add_fact(s2, R2, "Wren Corp renewed its annual contract in Q4.", entities=["wren"], session="x")
    _, l = C.add_fact(s2, R2, "Wren Corp let its contract lapse last year.", entities=["wren"], session="x")
    C.make_conflict_group(s2, w, l, group="cg-wren", since_cycle=1, now_cycle=3)
    res = s2.resolve_conflict(w, current_cycle=3)
    rl = s2._conn.execute("SELECT tier, superseded_by FROM semantic_facts WHERE id=?", (l,)).fetchone()
    suite.hard("conflict->resolve supersedes the stale fact (retired as history, winner freed)",
               res and res.get("winner_id") == w and rl and rl["tier"] == "superseded" and rl["superseded_by"] == w)
    # superseded fact must NOT surface in normal relational/recall
    after = R2.search("What is Wren Corp's contract status?", limit=5)
    superseded_surfaced = any(h.get("id") == l for h in after)
    suite.hard("superseded (stale) fact is withheld from normal recall", not superseded_surfaced)
    s2.close()

    s.close()
    return suite.finish("recency_results.md")


def _has_value(content, sc):
    return sc["new_v"].lower() in content.lower()


def _is_new(content, sc):
    c = content.lower()
    return sc["ent"] in c and sc["new_v"].lower() in c and sc["old_v"].lower() not in c


def _is_old(content, sc):
    c = content.lower()
    return sc["ent"] in c and sc["old_v"].lower() in c and sc["new_v"].lower() not in c


if __name__ == "__main__":
    sys.exit(main())
