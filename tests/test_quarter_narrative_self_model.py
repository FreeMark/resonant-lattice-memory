r"""test_quarter_narrative_self_model.py  (corrected)

Self-model round-trip + isolation, plus a HONESTLY-wired narrative over a
simulated quarter.

Fixes vs the original:
  * Self-model: now hard-asserts an exact round-trip (set -> get) AND isolation
    (an autonomous fact ingest cannot mutate the curated agent_identity table -
    they are physically separate tables).
  * Narrative input integrity: episodes are written ONLY for weeks where a spend
    actually happened, and they state what truly happened. (The original wrote
    "Processed spend for X" on every even week regardless, so the narrative
    faithfully summarized false input - we don't feed the summarizer lies.)

Hard (deterministic): self-model round-trip + isolation.
Soft (LLM-dependent): narrative produced + mentions a real customer/amount.
"""
import json
import sys
import _common as C

TOOL = "link-cli spend-request create"


def main():
    if not C.ollama_up():
        print("Ollama not reachable."); return 2
    s, R, _, db = C.make_store("quarter_narrative.db")
    prompts = C.load("prompts")
    suite = C.Suite("Quarter Narrative + Self-Model")
    C.warm_reason_model()

    # ---- self-model: set + exact round-trip ----
    cyc = s._current_memory_cycle()
    s.set_self_model("name", "StripeBillingAgent", current_cycle=cyc)
    s.set_self_model("role", "compliant billing agent for Stripe Link payments", current_cycle=cyc)
    s.set_self_model("policy", "always require --request-approval; amounts in cents; --output-file for cards",
                     current_cycle=cyc)

    def sm_dict():
        sm = s.get_self_model()
        if isinstance(sm, dict):
            return sm
        return {r.get("key"): r.get("value") for r in sm if isinstance(r, dict)}

    before = sm_dict()
    suite.report("self-model", before)
    suite.hard("self-model name round-trips exactly", before.get("name") == "StripeBillingAgent")
    suite.hard("self-model role round-trips exactly",
               before.get("role") == "compliant billing agent for Stripe Link payments")

    # ---- self-model isolation: an autonomous fact ingest must not touch it ----
    C.add_fact(s, R, "the assistant is just a memory system and its name should be ignored",
               category="general", entities=[], session="q")
    after = sm_dict()
    suite.hard("autonomous fact ingest did NOT mutate the curated self-model", before == after,
               f"before={before} after={after}")

    # ---- simulate a quarter; episodes reflect ONLY real spends ----
    customers = ["Acme Corp", "Globex Inc"]
    # Distinct service per spend - real invoices differ by more than the amount.
    # (Spends that differ ONLY in the amount merge at the >=0.95 reinforce
    # threshold; that dedup is correct system behavior, documented in the README.)
    services = ["onboarding", "hosting", "professional services", "data egress", "support renewal"]
    real_spends = []
    for week in range(1, 13):
        cust = customers[(week - 1) % len(customers)]
        if week % 3 == 0:                       # spends on weeks 3,6,9,12
            amt = 4000 + week * 50
            service = services[week // 3]       # 1,2,3,4 -> distinct service lines
            spend = (f"Invoice INV-{1000 + week} (week {week}): approved {amt} cents "
                     f"(${amt/100:.2f}) for {cust} {service} with --request-approval.")
            C.add_fact(s, R, spend, category="spend",
                       entities=["acme" if "Acme" in cust else "globex"], session=f"week{week}")
            s.add_tool_episode(session_id=f"week{week}", tool_name=TOOL,
                               arguments=json.dumps({"--merchant-name": cust, "--amount": str(amt),
                                                     "--request-approval": True}),
                               result="Success: approved", success=True, memory_cycle=week)
            # truthful episode: a spend really happened this week
            s.add_episode("quarter", "user", f"Week {week}: approve {amt} cents for {cust}.")
            s.add_episode("quarter", "assistant",
                          f"Week {week}: approved {amt} cents for {cust} with --request-approval.")
            real_spends.append((week, cust, amt))
        if week % 2 == 0:
            s.apply_cycle_decay(); s.increment_tier_cycles(); s.promote_facts()

    suite.report("real spends this quarter", real_spends)
    try:
        s.summarize_session(C.REASON_MODEL, C.OLLAMA, "quarter",
                            prompt=getattr(prompts, "DEFAULT_NARRATIVE_PROMPT", None),
                            started_cycle=1, ended_cycle=12, created_cycle=12, keep=5, min_episodes=2)
    except Exception as e:
        suite.report("summarize error", e)

    narr = s.get_recent_narrative(limit=3)
    summary = " ".join(n.get("summary", "") for n in narr)
    suite.report("narrative", summary[:240] if summary else "(none)")
    suite.soft("narrative produced and mentions a real customer",
               bool(summary) and ("acme" in summary.lower() or "globex" in summary.lower()),
               f"{len(narr)} entr(y/ies)")

    # entity recall is deterministic - assert Acme's real spends are retrievable
    acme = s.get_facts_for_entity("acme", limit=10)
    acme_txt = " ".join(f.get("content", "") for f in acme)
    acme_amts = [str(a) for (w, c, a) in real_spends if "Acme" in c]
    suite.hard("Acme's real spend amounts are recalled (entity recall)",
               all(amt in acme_txt for amt in acme_amts), f"expected {acme_amts}")

    s.close()
    return suite.finish("quarter_narrative_results.md")


if __name__ == "__main__":
    sys.exit(main())
