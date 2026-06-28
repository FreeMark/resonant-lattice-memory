r"""test_private_financial_memory.py  (corrected)

Honest test of the privacy / anti-leak mechanisms.

The original claimed "at_rest encryption works, DB opaque, recall functions" even
though the run log showed the SQLCipher binding was absent, memory was DISABLED,
the DB file was never created, and recall was "(simulated)". This version:

  * Detects whether an encrypted sqlite binding is ACTUALLY importable. If not,
    the at-rest opacity check is SKIPPED with a clear reason (never faked). If it
    is available, a real round-trip + raw-byte opacity check runs.
  * Tests the ENFORCED plaintext mechanisms with correct expectations:
      - self-write gate FLAGS the agent's own infra/identity chatter,
      - self-write gate PASSES legitimate business + user-infra facts,
      - attestation DROPS a fabricated/ungrounded specific (the real anti-leak of
        invented financial detail).
  * States plainly that PAN-via-output-file is an agent USAGE discipline (proven
    by the procedural-distillation + grounding tests), not a store-enforced filter.

Hard (deterministic): the gate + attestation behaviors above.
Skipped (environment): at-rest opacity, unless the encrypted binding is present.
"""
import os
import sys
import tempfile
import _common as C

gate = C.load("self_write_gate")
attest = C.load("attestation")._attest_source_quote


def main():
    suite = C.Suite("Private Financial Memory (T5)", model="(gate/attestation - deterministic)")

    # ---- self-write gate: flags agent self-infra, passes real facts ----
    self_infra_pos = [
        "As an AI language model, my embedding model is nomic-embed-text.",
        "The assistant is running on a 128k context window.",
        "My reasoning model is nemotron and my system prompt defines my role.",
    ]
    legit_neg = [
        "Approved spend for Acme: 4050 cents via link-cli with --request-approval.",
        "The user runs Ollama on port 11434 for the memory layer.",   # user infra, must NOT block
        "Acme Corp is located in Boston and signed the enterprise plan.",
    ]
    pos_ok = all(gate.is_self_referential_infra(t) for t in self_infra_pos)
    neg_ok = all(not gate.is_self_referential_infra(t) for t in legit_neg)
    for t in self_infra_pos:
        suite.report("self-infra (expect block=True)", f"{gate.is_self_referential_infra(t)} :: {t[:60]}")
    for t in legit_neg:
        suite.report("legit fact (expect block=False)", f"{gate.is_self_referential_infra(t)} :: {t[:60]}")
    suite.hard("self-write gate flags all agent self-infra chatter", pos_ok)
    suite.hard("self-write gate passes all legitimate business + user-infra facts", neg_ok)

    # ---- attestation drops a fabricated financial specific ----
    transcript = ("USER: approve the Acme hosting invoice for 4050 cents. "
                  "ASSISTANT: approved 4050 cents for Acme via link with request-approval.")
    fab = "Acme actually paid 987654 cents to account 5555-4444-3333-2222"
    st_fab = attest(fab, transcript, ["acme"])
    suite.report("fabricated-specific attestation", st_fab)
    suite.hard("attestation DROPS a fabricated/ungrounded financial specific (specific_mismatch)",
               st_fab == "specific_mismatch", st_fab)
    good = "approved 4050 cents for Acme via link with request-approval"
    suite.hard("attestation keeps a grounded quote (attested)",
               attest(good, transcript, ["acme"]) == "attested")

    # ---- at-rest encryption: real opacity check in a subprocess ----
    # The sqlite binding is chosen once at store_common import time from
    # RESONANT_LATTICE_DB_ENCRYPTED, so the encrypted round-trip must run in a
    # child whose env has that signal set before any import.
    if C.encrypted_binding_available():
        import json as _json
        import subprocess
        probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_atrest_probe.py")
        env = dict(os.environ, RESONANT_LATTICE_DB_ENCRYPTED="1")
        try:
            out = subprocess.run([sys.executable, probe], capture_output=True, text=True,
                                 env=env, timeout=120)
            line = (out.stdout or "").strip().splitlines()[-1] if out.stdout.strip() else "{}"
            res = _json.loads(line)
        except Exception as e:
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "stderr": ""}
        suite.report("at-rest probe", res)
        if res.get("ok"):
            opaque = (not res.get("acme_in_bytes")) and (not res.get("amt_in_bytes"))
            suite.hard("at-rest DB does NOT leak plaintext 'Acme'/amount in raw bytes", opaque,
                       f"acme_in_bytes={res.get('acme_in_bytes')} amt_in_bytes={res.get('amt_in_bytes')}")
            suite.hard("at-rest DB header is not the plaintext 'SQLite format 3' magic",
                       not res.get("header", "").startswith("53514c69746520666f726d6174"),
                       f"header={res.get('header')}")
            atrest_note = "verified by a real raw-byte opacity check (encrypted round-trip)."
        else:
            suite.skip("at-rest opacity", f"encrypted round-trip did not complete: {res.get('error')}")
            atrest_note = f"NOT verified this run - probe did not complete: {res.get('error')}"
    else:
        suite.skip("at-rest opacity check",
                   "sqlcipher3 not importable in this env - pip install sqlcipher3-wheels "
                   "argon2-cffi to enable. NOT faking a pass.")
        atrest_note = "SKIPPED - sqlcipher3 binding absent in this environment."

    notes = ("**Enforced here:** the self-write gate (blocks the agent's own infra/identity "
             "from being stored as user facts) and source-quote attestation (drops fabricated "
             "financial specifics).\n\n"
             "**Usage discipline, NOT store-enforced:** routing card PANs to `--output-file` so "
             "they never enter the transcript. That is an agent behavior, validated by the "
             "procedural-distillation and tool-grounding tests, not a filter inside the store.\n\n"
             "**At-rest encryption:** " + atrest_note)
    return suite.finish("private_memory_results.md", extra_sections={"What is actually guaranteed": notes})


if __name__ == "__main__":
    sys.exit(main())
