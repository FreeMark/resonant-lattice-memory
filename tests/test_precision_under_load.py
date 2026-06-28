r"""test_precision_under_load.py  (#1 - trust axis)

recall@k answers "is the right fact in the results?". This answers the harder,
money-relevant question: "is the TOP of the list trustworthy, or is the right
answer buried among irrelevant facts?".

Setup: T topics, each with R genuinely-relevant facts (sharing a rare topic
anchor) + a paraphrased query, mixed into thousands of irrelevant distractors.
For each query we measure:

  * precision@R   - of the top-R results, how many belong to the topic (ideal 1.0),
  * top-1 relevant- is the #1 result actually on-topic,
  * clean-cut     - do ALL R relevant facts rank above EVERY distractor (top-R is
                    exactly the relevant set),
  * relevance gap - separation between the weakest relevant hit and the strongest
                    distractor hit (a healthy index has a clear margin),
  * adaptive gate - does search(relevance_margin=...) drop distractors without
                    dropping relevant facts (the A6 precision gate).
"""
import os
import sys
import _common as C

LOAD = int(os.environ.get("RL_PREC_LOAD", "4000"))
R_PER_TOPIC = 5

# rare topic anchors (disjoint tokens) + R relevant fact bodies + a query each
TOPICS = [
    ("zephyrine-migration",
     ["The Zephyrine migration freezes writes during the Tromso cutover window.",
      "Zephyrine migration rollback is gated on a green checksum from the Ghent mirror.",
      "Only the Zephyrine migration lead may approve a schema change mid-cutover.",
      "Zephyrine migration progress is reported every six cycles to the steering group.",
      "The Zephyrine migration budget excludes the optional Almaty failover."],
     "What are the rules and status of the Zephyrine migration?"),
    ("tanager-billing",
     ["Tanager billing runs all charges through the shared payment token, never a raw PAN.",
      "Tanager billing caps a single auto-charge at 250000 cents before human review.",
      "Tanager billing reconciles against the Lisbon ledger on the last cycle of each month.",
      "A Tanager billing refund always requires a dual signature.",
      "Tanager billing suppresses duplicate invoices within a 3-cycle window."],
     "How does Tanager billing handle charges, caps, and refunds?"),
    ("kestrel-compliance",
     ["Kestrel compliance forbids storing customer card numbers in any note.",
      "Kestrel compliance mandates a Cusco-based review for spends over 1000000 cents.",
      "Kestrel compliance logs every approval with the approver and the cycle.",
      "Kestrel compliance treats a missing approval as an automatic rejection.",
      "Kestrel compliance audits the procedural rules every twelve cycles."],
     "What does Kestrel compliance require for approvals and card data?"),
    ("borealis-support",
     ["Borealis support guarantees a twelve-minute first response on tier-one tickets.",
      "Borealis support escalates an unacknowledged incident after two cycles.",
      "Borealis support routes data-loss reports straight to the platform on-call.",
      "Borealis support never promises a fix date without engineering sign-off.",
      "Borealis support closes a ticket only after the reporter confirms."],
     "What are the Borealis support response and escalation rules?"),
    ("orrery-contract",
     ["The Orrery contract auto-renews for one year unless cancelled 30 cycles prior.",
      "The Orrery contract pins pricing for the full term at signing.",
      "The Orrery contract requires written notice for any seat reduction.",
      "The Orrery contract excludes the Perth data-residency add-on by default.",
      "The Orrery contract caps annual uplift at five percent."],
     "What are the renewal and pricing terms of the Orrery contract?"),
    ("vantyx-procurement",
     ["Vantyx procurement needs three quotes for any purchase over 500000 cents.",
      "Vantyx procurement blocks a vendor lacking a signed data agreement.",
      "Vantyx procurement routes all approvals through the Osaka desk.",
      "Vantyx procurement records the purchase order against the originating team.",
      "Vantyx procurement disallows splitting an order to dodge the quote threshold."],
     "What are the Vantyx procurement approval and quote rules?"),
]


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    s, R, _, db = C.make_store("precision.db")
    suite = C.Suite("Precision Under Load (#1 trust)")

    # plant relevant facts
    relevant_ids = {}
    for anchor, facts, _q in TOPICS:
        ids = []
        for f in facts:
            a, fid = C.add_fact(s, R, f, category="topic", entities=[anchor], session="prec")
            ids.append(fid)
        relevant_ids[anchor] = set(ids)
        if len(set(ids)) != len(facts):
            suite.report(f"WARN: {anchor} had a merge", f"{len(set(ids))}/{len(facts)} distinct")

    # bury under load
    added = C.load_distractors(s, R, LOAD)
    for _ in range(2):
        s.increment_tier_cycles(); s.promote_facts()
    total = s._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    suite.report("distractors added", f"{added} of {LOAD}")
    suite.report("total rows in store", total)

    prec_at_r, top1_hits, clean_cuts, gaps = [], 0, 0, []
    for anchor, facts, query in TOPICS:
        rel = relevant_ids[anchor]
        hits = R.search(query, limit=R_PER_TOPIC)
        ids = [h.get("id") for h in hits]
        rel_in_topR = sum(1 for i in ids if i in rel)
        prec_at_r.append(rel_in_topR / R_PER_TOPIC)
        if ids and ids[0] in rel:
            top1_hits += 1
        if rel_in_topR == min(R_PER_TOPIC, len(rel)):
            clean_cuts += 1
        # relevance gap: weakest relevant vs strongest distractor in a top-20 pull
        wide = R.search(query, limit=20)
        rel_scores = [float(h.get("relevance", 0)) for h in wide if h.get("id") in rel]
        dis_scores = [float(h.get("relevance", 0)) for h in wide if h.get("id") not in rel]
        if rel_scores and dis_scores:
            gaps.append(round(min(rel_scores) - max(dis_scores), 3))
        suite.report(f"{anchor}", f"prec@{R_PER_TOPIC}={rel_in_topR}/{R_PER_TOPIC} top1={'rel' if ids and ids[0] in rel else 'DISTRACTOR'}")

    mean_prec = round(sum(prec_at_r) / len(prec_at_r), 3)
    suite.report("mean precision@5", mean_prec)
    suite.report("top-1 relevant", f"{top1_hits}/{len(TOPICS)}")
    suite.report("clean-cut (top-R == relevant set)", f"{clean_cuts}/{len(TOPICS)}")
    suite.report("relevance gaps (min_relevant - max_distractor)", gaps)

    suite.hard("top-1 is relevant for EVERY topic query (no distractor wins the #1 slot)",
               top1_hits == len(TOPICS), f"{top1_hits}/{len(TOPICS)}")
    suite.hard("mean precision@5 >= 0.8 under load", mean_prec >= 0.8, str(mean_prec))
    suite.soft("relevance gap is positive for every topic (relevant out-scores distractors)",
               all(g > 0 for g in gaps) if gaps else False, str(gaps))

    # ---- adaptive precision gate (A6): does relevance_margin drop the junk? ----
    gate_kept_rel, gate_dropped_dis, gate_dropped_rel = 0, 0, 0
    for anchor, facts, query in TOPICS:
        rel = relevant_ids[anchor]
        ungated = R.search(query, limit=20)
        gated = R.search(query, limit=20, relevance_margin=0.15)
        ug_ids = {h.get("id") for h in ungated}
        g_ids = {h.get("id") for h in gated}
        dropped = ug_ids - g_ids
        gate_dropped_dis += sum(1 for i in dropped if i not in rel)
        gate_dropped_rel += sum(1 for i in dropped if i in rel)
        gate_kept_rel += sum(1 for i in g_ids if i in rel)
    suite.report("adaptive gate: relevant kept / relevant dropped / distractors dropped",
                 f"{gate_kept_rel} / {gate_dropped_rel} / {gate_dropped_dis}")
    suite.soft("adaptive gate drops distractors without nuking relevant facts",
               gate_dropped_dis >= gate_dropped_rel, f"dropped_dis={gate_dropped_dis} dropped_rel={gate_dropped_rel}")

    s.close()
    return suite.finish("precision_results.md")


if __name__ == "__main__":
    sys.exit(main())
