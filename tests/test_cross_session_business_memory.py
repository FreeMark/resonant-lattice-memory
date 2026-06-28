r"""test_cross_session_business_memory.py  (corrected)

Closes the store after "session 1" and reopens a fresh instance on the SAME db
("session 2"), then verifies what actually persists across the restart.

Fixes vs the original:
  * Narrative: session 1 now actually CALLS summarize_session before closing
    (the original added episodes but never summarized, so narrative was always
    empty - then the report implied it worked).
  * Relational recall: tests a DETERMINISTIC relation (located_in) planted in
    session 1, instead of an (acme, approved, spend) triple the LLM never emits.

Hard (deterministic): entity-recall of the session-1 spend persists into session
2; tool episodes persist; the planted relation persists.
Soft (LLM-dependent): the session-1 narrative is present + mentions Acme.
"""
import json
import sys
import _common as C

TOOL = "link-cli spend-request create"


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    store_mod = C.load("store")
    retr_mod = C.load("retrieval")
    prompts = C.load("prompts")
    import tempfile, os
    db = os.path.join(tempfile.mkdtemp(), "cross_session.db")
    suite = C.Suite("Cross-Session Business Memory (T4)")
    C.warm_reason_model()

    def open_store():
        s = store_mod.LatticeStore(db_path=db, vector_dim=768, promotion_threshold=4,
                                   short_tier_cycles=1, mid_tier_cycles=1, initial_resonance=5)
        R = retr_mod.LatticeRetriever(s, C.OLLAMA, C.EMBED_MODEL, min_similarity=0.30)
        return s, R

    # ===== SESSION 1 =====
    s1, R1 = open_store()
    C.add_fact(s1, R1, "Acme Corp is located in Boston. Contact billing@acme.example.com.",
               category="customer", entities=["acme corp", "boston"], session="session-1",
               with_relations=True, use_llm=False)  # deterministic 'located in'
    spend1 = ("Approved spend for Acme Corp: $40.50 (4050 cents) hosting invoice via "
              "link-cli spend-request create --merchant-name \"Acme Corp\" --amount 4050 "
              "--request-approval --output-file /secure/acme_card.txt")
    C.add_fact(s1, R1, spend1, category="spend", entities=["acme corp"], session="session-1")
    s1.add_tool_episode(session_id="session-1", tool_name=TOOL,
                        arguments=json.dumps({"--merchant-name": "Acme Corp", "--amount": "4050",
                                              "--request-approval": True}),
                        result="Success: human approved in Link.", success=True,
                        memory_cycle=1, call_id="s1_001")
    # Drive a couple of episodes AND actually summarize the session (the fix).
    s1.add_episode("session-1", "user", "Onboard Acme Corp and approve the first $40.50 monthly spend.")
    s1.add_episode("session-1", "assistant", "Recorded Acme Corp details and approved 4050 cents via link with --request-approval.")
    s1.apply_cycle_decay(); s1.increment_tier_cycles(); s1.promote_facts()
    try:
        s1.summarize_session(C.REASON_MODEL, C.OLLAMA, "session-1",
                             prompt=getattr(prompts, "DEFAULT_NARRATIVE_PROMPT", None),
                             started_cycle=0, ended_cycle=1, created_cycle=1, keep=30, min_episodes=2)
    except Exception as e:
        suite.report("session-1 summarize error", e)
    s1.close()

    # ===== SESSION 2 (fresh instance, same db) =====
    s2, R2 = open_store()
    spend2 = ("Follow-up spend for Acme Corp: $52.50 (5250 cents) via link-cli spend-request create "
              "--merchant-name \"Acme Corp\" --amount 5250 --request-approval")
    C.add_fact(s2, R2, spend2, category="spend", entities=["acme corp"], session="session-2")
    s2.add_tool_episode(session_id="session-2", tool_name=TOOL,
                        arguments=json.dumps({"--merchant-name": "Acme Corp", "--amount": "5250",
                                              "--request-approval": True}),
                        result="Success", success=True, memory_cycle=2, call_id="s2_002")

    # ---- recall across the restart ----
    acme_facts = s2.get_facts_for_entity("acme corp", limit=10)
    contents = " ".join(f.get("content", "") for f in acme_facts)
    suite.report("acme facts visible in session 2", len(acme_facts))
    suite.hard("session-1 spend (4050/$40.50) recalled in session 2 via entity recall",
               "4050" in contents or "40.50" in contents)
    suite.hard("session-2 also sees the new 5250 spend", "5250" in contents)

    rr = s2.relational_recall(subject="acme corp", relation="located_in")
    suite.report("relational_recall(acme corp, located_in, ?)",
                 [(x["object"], x["match"]) for x in rr])
    suite.hard("planted relation (acme corp, located_in, boston) persists across sessions",
               any(x["object"] == "boston" for x in rr))

    eps = s2.get_recent_tool_episodes(tool_name=TOOL, limit=10)
    suite.report("tool episodes visible in session 2", len(eps))
    suite.hard("tool episodes from BOTH sessions persist", len(eps) >= 2,
               f"{len(eps)} episodes")

    narr = s2.get_recent_narrative(limit=3)
    summaries = " ".join(n.get("summary", "") for n in narr)
    suite.report("narrative entries in session 2", len(narr))
    suite.report("narrative text", summaries[:200] if summaries else "(none)")
    suite.soft("session-1 narrative persisted and mentions Acme",
               len(narr) >= 1 and ("acme" in summaries.lower()),
               f"{len(narr)} entr(y/ies)")

    s2.close()
    return suite.finish("cross_session_results.md")


if __name__ == "__main__":
    sys.exit(main())
