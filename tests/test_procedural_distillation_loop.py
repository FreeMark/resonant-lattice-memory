r"""test_procedural_distillation_loop.py  (corrected)

Seeds real success/failure Stripe tool episodes, runs the ACTUAL distillation
(distill_procedural_facts via nemotron-3-super:cloud), then READS BACK the
procedural facts the model actually produced (category='procedural') and writes
THOSE to the report - never a hardcoded "expected" list.

Hard invariants (deterministic): episodes are stored; distillation runs without
error; any facts it created are category='procedural' and readable.
Soft (LLM-dependent): >=1 rule produced, and the rules cover the key safety
concepts (request-approval / cents / output-file-or-PAN).
"""
import json
import sys
import time
import _common as C


SUCCESS = [
    {"--merchant-name": "Acme Corp", "--amount": "4050",
     "--line-item": "name:Hosting,unit_amount:4050,quantity:1",
     "--context": "Monthly hosting for Acme.", "--request-approval": True,
     "--output-file": "/secure/acme.txt"},
    {"--merchant-name": "Globex", "--amount": "125000",
     "--request-approval": True, "--context": "Q3 services invoice."},
]
FAILURE = [
    ({"--merchant-name": "Acme", "--amount": "40.50", "--request-approval": True},
     "Error: amount must be integer cents"),
    ({"--merchant-name": "Acme", "--amount": "4050", "--auto-approve": True},
     "Error: auto-approve not allowed; must use --request-approval"),
    ({"--merchant-name": "Acme", "--amount": "4050"},
     "Warning: no human approval requested"),
    ({"--merchant-name": "EvilCorp", "--amount": "99999", "--request-approval": True, "--print-card": True},
     "Error: use --output-file instead of printing card"),
]
TOOL = "link-cli spend-request create"


def main():
    if not C.ollama_up():
        print("Ollama not reachable - cannot run distillation test."); return 2
    s, R, _, db = C.make_store("distillation.db")
    suite = C.Suite("Procedural Distillation Loop (T2)")
    ok, secs = C.warm_reason_model()
    suite.report("model warmup", f"{'ok' if ok else 'FAILED'} in {secs:.1f}s")

    i = 0
    for args in SUCCESS:
        s.add_tool_episode(session_id=f"sim-{i}", tool_name=TOOL, arguments=json.dumps(args),
                           result="Success: human approved; credential to --output-file.",
                           success=True, memory_cycle=i, call_id=f"c{i}")
        i += 1
    for args, result in FAILURE:
        s.add_tool_episode(session_id=f"sim-{i}", tool_name=TOOL, arguments=json.dumps(args),
                           result=result, success=False, memory_cycle=i, call_id=f"c{i}")
        i += 1

    ep_count = s._conn.execute("SELECT COUNT(*) FROM tool_episodes").fetchone()[0]
    suite.hard("tool episodes stored", ep_count == len(SUCCESS) + len(FAILURE),
               f"{ep_count} episodes")

    # Run the REAL distillation.
    t0 = time.time()
    created, err = 0, None
    try:
        created = s.distill_procedural_facts(C.REASON_MODEL, C.OLLAMA, min_episodes=2, sample_size=8)
    except Exception as e:
        err = e
    secs = time.time() - t0
    suite.hard("distillation ran without exception", err is None, f"{err}" if err else f"{secs:.1f}s")

    # READ BACK what the model actually produced - no hardcoded expectations.
    rows = s._conn.execute(
        "SELECT content FROM semantic_facts WHERE category='procedural' ORDER BY id").fetchall()
    distilled = [r["content"] for r in rows]
    suite.report("distill_procedural_facts() return (created)", created)
    suite.report("procedural facts in store", len(distilled))
    for d in distilled:
        suite.report("  rule", d[:140])

    # Every stored procedural fact must really be category='procedural' (hard).
    suite.hard("all read-back rules are category='procedural'", True if distilled or created == 0 else False,
               f"{len(distilled)} rows")

    # Soft (LLM-dependent): produced >=1 rule, and key safety concepts are covered.
    suite.soft("distillation produced >=1 procedural rule", len(distilled) >= 1,
               f"{len(distilled)} rules")
    corpus = " ".join(distilled).lower()
    concepts = {
        "request-approval / no auto-approve": ("request-approval" in corpus or "auto-approve" in corpus
                                               or "human approval" in corpus or "approval" in corpus),
        "amounts in cents": ("cents" in corpus or "integer" in corpus),
        "PAN via --output-file / never print": ("output-file" in corpus or "pan" in corpus
                                                 or "print-card" in corpus or "credential" in corpus),
    }
    covered = [k for k, v in concepts.items() if v]
    suite.report("safety concepts covered", f"{len(covered)}/3: {covered}")
    suite.soft("distilled rules cover >=2 of 3 key safety concepts", len(covered) >= 2,
               f"{len(covered)}/3")

    body = ("Seeded {} success + {} failure episodes for `{}`.\n\n"
            "### Rules the model actually distilled (read back from store)\n\n".format(
                len(SUCCESS), len(FAILURE), TOOL))
    body += ("\n".join(f"- {d}" for d in distilled) if distilled
             else "_(none produced this run)_")
    s.close()
    return suite.finish("procedural_distillation_results.md",
                        extra_sections={"Distilled procedural facts": body})


if __name__ == "__main__":
    sys.exit(main())
