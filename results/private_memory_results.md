# Private Financial Memory (T5)

**Date**: 2026-06-25 14:26:30  
**Memory model**: (gate/attestation - deterministic)  
**Embed model**: nomic-embed-text  

**Verdict (hard invariants)**: PASS (6/6)  

## Checks (measured)

| kind | check | status | detail |
|---|---|---|---|
| info | self-infra (expect block=True) | INFO | True :: As an AI language model, my embedding model is nomic-embed-t |
| info | self-infra (expect block=True) | INFO | True :: The assistant is running on a 128k context window. |
| info | self-infra (expect block=True) | INFO | True :: My reasoning model is nemotron and my system prompt defines  |
| info | legit fact (expect block=False) | INFO | False :: Approved spend for Acme: 4050 cents via link-cli with --requ |
| info | legit fact (expect block=False) | INFO | False :: The user runs Ollama on port 11434 for the memory layer. |
| info | legit fact (expect block=False) | INFO | False :: Acme Corp is located in Boston and signed the enterprise pla |
| hard | self-write gate flags all agent self-infra chatter | PASS |  |
| hard | self-write gate passes all legitimate business + user-infra facts | PASS |  |
| info | fabricated-specific attestation | INFO | specific_mismatch |
| hard | attestation DROPS a fabricated/ungrounded financial specific (specific_mismatch) | PASS | specific_mismatch |
| hard | attestation keeps a grounded quote (attested) | PASS |  |
| info | at-rest probe | INFO | {'ok': True, 'size': 241664, 'acme_in_bytes': False, 'amt_in_bytes': False, 'header': '8885f149340d2f8d2f109c06e749280f'} |
| hard | at-rest DB does NOT leak plaintext 'Acme'/amount in raw bytes | PASS | acme_in_bytes=False amt_in_bytes=False |
| hard | at-rest DB header is not the plaintext 'SQLite format 3' magic | PASS | header=8885f149340d2f8d2f109c06e749280f |

## What is actually guaranteed

**Enforced here:** the self-write gate (blocks the agent's own infra/identity from being stored as user facts) and source-quote attestation (drops fabricated financial specifics).

**Usage discipline, NOT store-enforced:** routing card PANs to `--output-file` so they never enter the transcript. That is an agent behavior, validated by the procedural-distillation and tool-grounding tests, not a filter inside the store.

**At-rest encryption:** verified by a real raw-byte opacity check (encrypted round-trip).
