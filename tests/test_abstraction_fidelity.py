r"""test_abstraction_fidelity.py  (does COMPRESSION preserve exact $ / IDs?)

A money agent dies on rounding: if memory compresses "4050 cents" into "about
$40" or invents "$4000", the agent acts on a wrong number. Two compressors:

  ABSTRACTION (perform_abstraction_pass): clusters related facts into a general
    'abstract' fact. It ADDS a fact (sources remain), and the prompt forbids
    inventing generalizations — so the trust bar is: it must never introduce a
    number that isn't in the sources (no fabricated/rounded value). Exact values
    stay recoverable from the surviving source rows.

  GIST (consolidate_before_prune): summarizes DYING facts right before deletion.
    By default the prompt says to DROP exact numbers — so a gist-then-prune of a
    money fact LOSES the exact value (the only surviving record is blurred). For
    money/compliance facts the number IS the meaning; this probe measures that
    loss (and, after the prompt fix, confirms the value survives).

Hard: abstraction introduces NO fabricated number; sources survive. Gist:
measured (drives the prompt fix) -> after fix, exact values survive the gist.
"""
import os
import re
import sys
import _common as C

REASON = C.REASON_MODEL


def nums(text):
    return {re.sub(r"\D", "", m) for m in re.findall(r"\d[\d,]*", text or "")
            if len(re.sub(r"\D", "", m)) >= 3}   # >=3 digits = a real amount/ID, not "1 of 2"


def main():
    if not C.ollama_up():
        print("Ollama not reachable."); return 2
    prompts = C.load("prompts")
    suite = C.Suite("Abstraction / Gist Fidelity (exact $ + IDs)")
    C.warm_reason_model()

    # ---- A: ABSTRACTION fidelity ----
    sa, Ra, _, _ = C.make_store("absfid.db")
    SRC = [
        "Acme invoice INV-4001 was 4050 cents for hosting.",
        "Acme invoice INV-4002 was 9900 cents for support.",
        "Acme invoice INV-4003 was 12500 cents for data egress.",
        "Acme invoice INV-4004 was 33000 cents for onboarding.",
        "Acme invoice INV-4005 was 7800 cents for overage.",
    ]
    src_nums = set()
    for c in SRC:
        C.add_fact(sa, Ra, c, category="spend", entities=["acme"], session="a")
        src_nums |= nums(c)
    n_src = sa._conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
    suite.hard("source invoice facts stored distinctly (merge fix holds)", n_src == len(SRC), f"{n_src}/{len(SRC)}")
    for c in range(1, 3):
        sa.set_cycle_counts(memory_cycle=c); sa.increment_tier_cycles(); sa.promote_facts()
    try:
        sa.perform_abstraction_pass(REASON, C.OLLAMA, prompt=prompts.DEFAULT_CONSOLIDATION_PROMPT,
                                    min_cluster_size=2, cluster_entity_overlap=0.3,
                                    cluster_hrr_similarity=0.5, max_clusters=3)
    except Exception as e:
        suite.report("abstraction note", e)
    abstracts = [r["content"] for r in sa._conn.execute(
        "SELECT content FROM semantic_facts WHERE category='abstract'").fetchall()]
    suite.report("abstractions created", len(abstracts))
    for a in abstracts:
        suite.report("  abstract", a[:130])
    abs_nums = set()
    for a in abstracts:
        abs_nums |= nums(a)
    fabricated = abs_nums - src_nums
    suite.hard("abstraction introduces NO fabricated/rounded number (all numbers trace to sources)",
               not fabricated, f"fabricated={sorted(fabricated)}")
    still = sa._conn.execute("SELECT COUNT(*) FROM semantic_facts WHERE category='spend'").fetchone()[0]
    suite.hard("source facts survive abstraction (exact values stay recoverable)", still == len(SRC),
               f"{still}/{len(SRC)} sources intact")
    suite.report("exact source amounts echoed in the abstraction",
                 f"{len(abs_nums & src_nums)}/{len(src_nums)}")
    sa.close()

    # ---- B: GIST fidelity (dying money facts -> gist -> is the value kept?) ----
    sb, Rb, _, _ = C.make_store("gistfid.db", initial_resonance=6)
    GIST_SRC = [
        "Acme paid 4050 cents on invoice INV-7001 for the March hosting cycle.",
        "Acme paid 9900 cents on invoice INV-7002 for the March support cycle.",
        "Acme paid 12500 cents on invoice INV-7003 for the March egress cycle.",
    ]
    gist_nums = set()
    gids = []
    for c in GIST_SRC:
        _, fid = C.add_fact(sb, Rb, c, category="spend", entities=["acme"], session="b")
        gids.append(fid); gist_nums |= nums(c)
    for c in range(1, 3):
        sb.set_cycle_counts(memory_cycle=c); sb.increment_tier_cycles(); sb.promote_facts()
    for fid in gids:                                  # drive them DYING (peak stays high)
        sb.adjust_resonance(fid, -100)
    sb._conn.commit()
    try:
        sb.consolidate_before_prune(REASON, C.OLLAMA, prompt=prompts.DEFAULT_GIST_PROMPT,
                                    gist_floor=0.0, min_peak_resonance=4.0,
                                    cluster_hrr_similarity=0.5, cluster_entity_overlap=0.3,
                                    min_cluster_size=2, max_clusters=2)
    except Exception as e:
        suite.report("gist note", e)
    gists = [r["content"] for r in sb._conn.execute(
        "SELECT content FROM semantic_facts WHERE category='gist'").fetchall()]
    suite.report("gists created", len(gists))
    for g in gists:
        suite.report("  gist", g[:140])
    g_nums = set()
    for g in gists:
        g_nums |= nums(g)
    suite.hard("gist introduces NO fabricated number", not (g_nums - gist_nums),
               f"fabricated={sorted(g_nums - gist_nums)}")
    preserved = len(g_nums & gist_nums)
    suite.report("exact amounts preserved by the gist", f"{preserved}/{len(gist_nums)}")
    # The fidelity bar: a gist of money facts must keep the exact amounts (so when
    # the source is pruned, the value is not lost to rounding).
    suite.hard("gist preserves the exact money amounts (>=1, ideally all)",
               preserved >= 1 if gists else False, f"{preserved}/{len(gist_nums)}")
    sb.close()

    return suite.finish("abstraction_fidelity_results.md")


if __name__ == "__main__":
    sys.exit(main())
