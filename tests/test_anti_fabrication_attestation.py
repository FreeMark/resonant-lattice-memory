r"""test_anti_fabrication_attestation.py  (corrected)

Tests the ENFORCED anti-fabrication mechanisms with assertions that actually
exercise them:

  1. Attestation DROP path: a fact whose content carries a hard specific
     (number/id) that is ABSENT from its own source_quote returns
     'specific_mismatch'. (The original test put the fabricated number inside the
     quote, so attestation correctly could not flag it - that proved nothing.)
  2. Attestation KEEP path: a fact whose content is grounded by its quote
     returns 'attested' (or 'soft').
  3. get_fact returns the EXACT stored content (no phantom mutation).
  4. Conflict machinery: place two facts in a conflict group (as
     resolve_hrr_conflicts would) and assert pending_conflicts surfaces them and
     resolve_conflict supersedes the loser + boosts the winner.

Also REPORTS (honestly) whether organic HRR conflict detection fires on two
near-identical financial facts - it usually does not, because >0.90 similarity is
treated as reinforcement/duplicate, not a dispute. That is real, documented
behavior, not a pass/fail.
"""
import sys
import _common as C

attest = C.load("attestation")._attest_source_quote


def main():
    if not C.ollama_up():
        print("Ollama not reachable (embeddings needed)."); return 2
    s, R, _, db = C.make_store("anti_fab.db")
    suite = C.Suite("Anti-Fabrication + Source-Quote Attestation (T3)")

    # ---- 1 & 2: attestation drop vs keep (deterministic; no LLM) ----
    # _attest_source_quote(claimed_quote, full_transcript, entities): does the
    # quote's hard specifics actually appear in the source transcript?
    transcript = ("USER: please approve the Acme hosting invoice for 4050 cents. "
                  "ASSISTANT: approved 4050 cents for Acme via link with request-approval; "
                  "credential written to the output file, no card number in the log.")
    grounded_quote = "approved 4050 cents for Acme via link with request-approval"
    st_grounded = attest(grounded_quote, transcript, ["acme"])
    suite.report("grounded quote attestation", st_grounded)
    suite.hard("grounded quote is attested (its 4050/acme appear in transcript)",
               st_grounded == "attested", st_grounded)

    fabricated_quote = "Acme paid exactly 987654 cents, tx TX-FAKE-123"
    st_fab = attest(fabricated_quote, transcript, ["acme"])
    suite.report("fabricated (ungrounded number) attestation", st_fab)
    suite.hard("fabricated hard-number quote is flagged specific_mismatch (DROP)",
               st_fab == "specific_mismatch", st_fab)

    # ---- 3: exact stored value, no phantom mutation ----
    grounded_content = "Approved spend for Acme of 4050 cents, human approved via Link."
    _, fid_good = C.add_fact(s, R, grounded_content, category="spend", entities=["acme"],
                             source_quote=grounded_quote, session="af")
    got = s.get_fact(fid_good)
    suite.hard("get_fact returns the exact stored content",
               got and got["content"] == grounded_content)
    suite.hard("stored amount 4050 present and not mutated into a phantom value",
               got and "4050" in got["content"] and "987654" not in got["content"])

    # ---- 4: conflict machinery (deterministic) ----
    # Phrasings are deliberately divergent so they do NOT merge on insert
    # (near-identical text >=0.95 cosine reinforces into one row instead).
    aw, w = C.add_fact(s, R, "Finance confirms Acme's approved June hosting charge was 4050 cents.",
                       entities=["acme"], session="af")
    al, l = C.add_fact(s, R, "A vendor-portal discrepancy reports the Acme June figure differently, as 4500 cents.",
                       entities=["acme"], session="af")
    suite.hard("two distinct conflicting facts exist (both added as separate rows)",
               aw == "added" and al == "added" and w != l, f"aw={aw} w={w} | al={al} l={l}")

    C.make_conflict_group(s, w, l, group="cg-amt", since_cycle=1, now_cycle=3)
    pend = s.get_pending_conflicts(min_age_cycles=0)
    suite.hard("pending_conflicts surfaces the disputed group",
               len(pend) == 1 and pend[0]["conflict_group_id"] == "cg-amt",
               f"{len(pend)} group(s)")
    suite.hard("age gate hides a too-young group",
               s.get_pending_conflicts(min_age_cycles=5) == [])
    res = s.resolve_conflict(w, current_cycle=3)
    suite.hard("resolve_conflict picks the winner + supersedes the loser",
               res and res.get("winner_id") == w and l in res.get("superseded", []), res)
    rl = s._conn.execute("SELECT tier, superseded_by FROM semantic_facts WHERE id=?", (l,)).fetchone()
    suite.hard("loser retired as superseded history (not deleted)",
               rl and rl["tier"] == "superseded" and rl["superseded_by"] == w, dict(rl) if rl else None)
    suite.hard("group resolved (no longer pending)", s.get_pending_conflicts(min_age_cycles=0) == [])

    # ---- organic HRR conflict detection (now ROBUST) ----
    # Before the mid+long / overlap-coefficient fix this fired 0 groups (the pair
    # was short-tier and Jaccard 0.33 excluded the attribute contradiction). It
    # now reliably detects it; the content-similarity band still discriminates.
    s2, R2, _, _ = C.make_store("anti_fab_organic.db")
    _, a = C.add_fact(s2, R2, "user lives in Seattle", entities=["user", "seattle"], session="o")
    _, b = C.add_fact(s2, R2, "user lives in Portland", entities=["user", "portland"], session="o")
    suite.hard("two distinct location facts (not merged on insert)", a != b, f"a={a} b={b}")
    s2.increment_tier_cycles(); s2.promote_facts()    # short -> mid so the scan sees them
    s2.resolve_hrr_conflicts()
    suite.hard("organic HRR detection fires on the 'lives in Seattle/Portland' attribute contradiction",
               len(s2.get_pending_conflicts(min_age_cycles=0)) == 1)
    s2.close()

    # negative control: unrelated facts must NOT be forced into a conflict
    s3, R3, _, _ = C.make_store("anti_fab_negctrl.db")
    C.add_fact(s3, R3, "Acme is headquartered in Boston", entities=["acme", "boston"], session="n")
    C.add_fact(s3, R3, "Globex grew its revenue last quarter", entities=["globex"], session="n")
    s3.increment_tier_cycles(); s3.promote_facts()
    s3.resolve_hrr_conflicts()
    suite.hard("no false-positive conflict between unrelated facts",
               s3.get_pending_conflicts(min_age_cycles=0) == [])
    s3.close()

    s.close()
    return suite.finish("anti_fabrication_results.md")


if __name__ == "__main__":
    sys.exit(main())
