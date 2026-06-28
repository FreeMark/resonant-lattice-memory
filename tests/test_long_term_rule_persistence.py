r"""test_long_term_rule_persistence.py  (corrected)

Verifies the PINNING invariant under sustained aggressive decay:
  * pinned compliance rules are protected from decay/prune (stay high + 'long'),
  * AND unpinned operational facts genuinely fade toward 0.

Both halves are required: if nothing decays, "pinning protected them" proves
nothing. This is a pure-substrate test (no LLM needed).
"""
import sys
import _common as C


def main():
    s, R, _, db = C.make_store("long_term_pinning.db", decay_per_cycle=0.8)
    suite = C.Suite("Long-Term Rule Persistence + Pinning", model="(no LLM - substrate only)")

    facts = [
        ("Customer Acme Corp prefers Net-30 payment terms.", "customer", ["acme"], False),
        ("Last Q3 revenue from Globex was $125,000.", "financial", ["globex"], False),
        ("Invoice #4821 for $4,050 was approved on 2025-03-15.", "spend", ["4821"], False),
        ("POLICY: Never set auto_approve=true on any spend over $1000.", "policy", [], True),
        ("POLICY: Payments above $500 require explicit human approval via Link app.", "policy", [], True),
        ("POLICY: All amounts MUST be recorded in cents (e.g. $40.50 = 4050).", "policy", [], True),
    ]
    pinned_ids, normal_ids = [], []
    for content, cat, ents, pin in facts:
        _, fid = C.add_fact(s, R, content, category=cat, entities=ents, session="biz")
        if pin:
            s.set_pinned(fid, True)
            pinned_ids.append(fid)
        else:
            normal_ids.append(fid)

    suite.hard("all 6 facts persisted on insert",
               len(pinned_ids) == 3 and len(normal_ids) == 3 and all(f > 0 for f in pinned_ids + normal_ids))

    NUM_CYCLES = 80
    for _ in range(NUM_CYCLES):
        s.apply_cycle_decay(protect_conflicts=False)
        s.increment_tier_cycles()
        s.promote_facts()
        for row in s._conn.execute("SELECT id FROM semantic_facts WHERE COALESCE(pinned,0)=0"):
            s.adjust_resonance(row[0], -1)   # extra pressure on unpinned only

    def state(fid):
        r = s.get_fact(fid)
        return (None, None, None) if not r else (r.get("resonance_count"), r.get("tier"), bool(r.get("pinned")))

    pinned_state = {fid: state(fid) for fid in pinned_ids}
    normal_state = {fid: state(fid) for fid in normal_ids}
    suite.report("pinned final states", pinned_state)
    suite.report("normal final states", normal_state)

    # Pinned: survived (still present), still flagged, still 'long', resonance not bled.
    pinned_alive = [f for f, (res, tier, pin) in pinned_state.items()
                    if res is not None and pin and tier == "long" and res > 2.0]
    suite.hard(f"all {len(pinned_ids)} pinned policies protected (present, pinned, long, res>2)",
               len(pinned_alive) == len(pinned_ids),
               f"{len(pinned_alive)}/{len(pinned_ids)} protected")

    # Normal: actually decayed (proves the protection above is meaningful).
    # A row may be deleted (state None) or driven near zero - both count as faded.
    normal_faded = [f for f, (res, tier, pin) in normal_state.items()
                    if res is None or res <= 1.0]
    suite.hard(f"all {len(normal_ids)} unpinned facts faded after {NUM_CYCLES} cycles (res<=1 or pruned)",
               len(normal_faded) == len(normal_ids),
               f"{len(normal_faded)}/{len(normal_ids)} faded")

    s.close()
    return suite.finish("long_term_pinning_results.md")


if __name__ == "__main__":
    sys.exit(main())
