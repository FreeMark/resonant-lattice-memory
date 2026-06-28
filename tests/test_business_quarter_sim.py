r"""test_business_quarter_sim.py  (corrected)

Simulates ~12 weeks of sales + spend with periodic dream cycles, then verifies
long-term robustness with real substrate reads.

Fixes vs the original:
  * Narrative is wired correctly: episodes are appended to the SAME session that
    is later summarized (the original summarized "business-q", a session that had
    no episodes, so narrative was always empty).
  * Hard asserts replace the prose "Strong" conclusion: pinned rules survive,
    every recorded spend is recalled, and no phantom amount appears.

Hard (deterministic): pinned survival, spend recall, no-phantom.
Soft (LLM-dependent): narrative captured.
"""
import sys
import _common as C

SESSION = "biz-quarter"


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    s, R, _, db = C.make_store("business_quarter.db",
                               short_tier_cycles=2, mid_tier_cycles=4, decay_per_cycle=0.4)
    prompts = C.load("prompts")
    suite = C.Suite("Business Quarter Simulator")
    C.warm_reason_model()

    # pinned compliance rules
    policy = [
        "RULE: Never auto-approve any spend > $1000. Always set --request-approval.",
        "RULE: Record every amount in cents only. $40.50 must be 4050.",
        "RULE: All vendor payments require documented human approval in Link app.",
    ]
    pinned_ids = []
    for rule in policy:
        _, fid = C.add_fact(s, R, rule, category="policy", entities=[], session=SESSION)
        s.set_pinned(fid, True)
        pinned_ids.append(fid)

    customers = ["Acme Corp", "Globex Inc", "Stark Industries"]
    recorded = []   # (week, customer, amount_cents)
    for week in range(1, 13):
        cust = customers[(week - 1) % len(customers)]
        ent = "acme" if "Acme" in cust else ("globex" if "Globex" in cust else "stark")
        if week % 2 == 1:
            C.add_fact(s, R, f"Week {week}: sales call with {cust}; interested in enterprise plan.",
                       category="sales", entities=[ent], session=SESSION)
        if week % 3 == 0:
            amt = 4050 + week * 100
            # Unique invoice id -> distinct fact (templated near-identical spend
            # strings otherwise merge at the >=0.95 reinforce threshold).
            C.add_fact(s, R, f"Invoice INV-{2000 + week} (week {week}): approved {amt} cents for {cust} hosting; --request-approval.",
                       category="spend", entities=[ent], session=SESSION)
            recorded.append((week, cust, amt))
            # truthful narrative episodes, same session that we later summarize
            s.add_episode(SESSION, "user", f"Week {week}: approve {amt} cents for {cust}.")
            s.add_episode(SESSION, "assistant", f"Week {week}: approved {amt} cents for {cust} with --request-approval.")
        if week % 2 == 0:
            s.apply_cycle_decay(); s.increment_tier_cycles(); s.promote_facts()

    # quarter-end consolidation + narrative
    for _ in range(2):
        s.apply_cycle_decay(); s.increment_tier_cycles(); s.promote_facts()
    try:
        s.summarize_session(C.REASON_MODEL, C.OLLAMA, SESSION,
                            prompt=getattr(prompts, "DEFAULT_NARRATIVE_PROMPT", None),
                            started_cycle=1, ended_cycle=12, created_cycle=12, keep=10, min_episodes=2)
    except Exception as e:
        suite.report("summarize error", e)

    # ---- robustness checks ----
    pinned_now = [fid for fid in pinned_ids if (s.get_fact(fid) or {}).get("pinned")]
    suite.hard("all pinned compliance rules survived the quarter",
               len(pinned_now) == len(pinned_ids), f"{len(pinned_now)}/{len(pinned_ids)}")

    recalled = 0
    for week, cust, amt in recorded:
        ent = "acme" if "Acme" in cust else ("globex" if "Globex" in cust else "stark")
        facts = s.get_facts_for_entity(ent, limit=10)
        if any(str(amt) in f.get("content", "") for f in facts):
            recalled += 1
    suite.report("recorded spends", recorded)
    suite.hard("every recorded spend is recalled by entity at quarter end",
               recalled == len(recorded), f"{recalled}/{len(recorded)}")

    phantom = any("99999" in row[0] for row in s._conn.execute("SELECT content FROM semantic_facts"))
    suite.hard("no phantom/fabricated amount present", not phantom)

    tiers = {r["tier"]: r["cnt"] for r in s._conn.execute(
        "SELECT tier, COUNT(*) cnt FROM semantic_facts GROUP BY tier").fetchall()}
    suite.report("tier distribution", tiers)

    narr = s.get_recent_narrative(limit=3)
    summary = " ".join(n.get("summary", "") for n in narr)
    suite.report("narrative", summary[:240] if summary else "(none)")
    suite.soft("narrative captured business activity (mentions a customer)",
               bool(summary) and any(c.split()[0].lower() in summary.lower() for c in customers),
               f"{len(narr)} entr(y/ies)")

    caveat = ("Spend facts use a unique invoice id so each is a distinct row. "
              "Near-identical templated spend strings that differ ONLY in the amount "
              "merge at the >=0.95 reinforce threshold (one row, reinforced) - a real "
              "consideration for high-volume templated financial logs: include a "
              "distinguishing token (invoice id / date) or pin facts that must persist verbatim.")
    s.close()
    return suite.finish("business_quarter_results.md",
                        extra_sections={"Caveat: near-identical fact merging": caveat})


if __name__ == "__main__":
    sys.exit(main())
