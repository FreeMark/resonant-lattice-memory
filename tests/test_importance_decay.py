r"""test_importance_decay.py  (feature: importance != frequency)

The fade probe showed retention depends on use/pin/novelty — so an important-but-
rarely-recalled fact (a compliance rule, a spend record) can fade like noise. The
importance_decay_discount feature makes high-stakes-category facts decay slower,
so they survive the dwell-to-long gauntlet and are retained where generic facts
prune. This test proves: with the feature ON, an UNUSED, UNPINNED important fact
is retained while a matched generic fact fades; with it OFF, both fade (so the
feature is what saved it).

Config: recall-required-ish regime (aggressive decay, novelty OFF) so a generic
unused fact reliably fades, isolating the importance effect.
"""
import os
import sys
import _common as C

IMPORTANT = [
    ("POLICY: vendor refunds require a logged second-approver signature.", "compliance"),
    ("Acme committed annual contract value is 4500000 cents for FY25.", "financial"),
    ("Spend approvals above 1000000 cents must route through the Osaka desk.", "policy"),
]
GENERIC = [
    ("The team tried the new cafe near the Lisbon office last week.", "general"),
    ("Someone mentioned the weather was nice during the Tromso trip.", "general"),
    ("A casual note about rearranging the standup time once.", "general"),
]
CYCLES = 30


def retain_run(importance_discount):
    """Run CYCLES of decay with the given importance_discount; return which
    important / generic fact ids are still present (not pruned) + recallable."""
    s, R, _, db = C.make_store("impdecay.db",
                               initial_resonance=4, promotion_threshold=4,
                               decay_per_cycle=1.0, novelty_enabled=False,
                               short_tier_cycles=2, mid_tier_cycles=3)
    imp_ids, gen_ids = [], []
    for content, cat in IMPORTANT:
        _, fid = C.add_fact(s, R, content, category=cat, entities=[], session="i"); imp_ids.append(fid)
    for content, cat in GENERIC:
        _, fid = C.add_fact(s, R, content, category=cat, entities=[], session="g"); gen_ids.append(fid)
    for c in range(1, CYCLES + 1):
        s.set_cycle_counts(memory_cycle=c)
        s.apply_cycle_decay(importance_discount=importance_discount)   # the lever
        s.increment_tier_cycles(); s.promote_facts()
        s.prune_weak_facts(forget_after_cycles=4)
    def present(ids):
        return sum(1 for i in ids
                   if s._conn.execute("SELECT COUNT(*) FROM semantic_facts WHERE id=?", (i,)).fetchone()[0])
    imp_present, gen_present = present(imp_ids), present(gen_ids)
    s.close()
    return imp_present, gen_present


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings)."); return 2
    suite = C.Suite("Importance-Weighted Decay (importance != frequency)", model="(no LLM)")

    # OFF (control): both important + generic should fade (nothing pins/uses them)
    imp_off, gen_off = retain_run(0.0)
    suite.report("feature OFF — important retained / generic retained", f"{imp_off}/{len(IMPORTANT)}  {gen_off}/{len(GENERIC)}")
    suite.hard("control: with the feature OFF, unused important facts fade like generic ones",
               imp_off == 0 and gen_off == 0, f"imp={imp_off} gen={gen_off}")

    # ON: important facts retained, generic still fade
    imp_on, gen_on = retain_run(0.6)
    suite.report("feature ON (0.6) — important retained / generic retained", f"{imp_on}/{len(IMPORTANT)}  {gen_on}/{len(GENERIC)}")
    suite.hard("with the feature ON, UNUSED important facts are RETAINED (importance != frequency)",
               imp_on == len(IMPORTANT), f"{imp_on}/{len(IMPORTANT)}")
    suite.hard("the feature is SELECTIVE — generic noise still fades (no unbounded retention)",
               gen_on == 0, f"generic retained={gen_on}")
    suite.hard("ON retains strictly more important facts than OFF (the feature is the cause)",
               imp_on > imp_off, f"off={imp_off} on={imp_on}")

    return suite.finish("importance_decay_results.md")


if __name__ == "__main__":
    sys.exit(main())
