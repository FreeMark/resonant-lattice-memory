# RLM memory - operating rule

> Installed by the RLM<->grok integration. This is NOT your `AGENTS.md` project kit; it only
> tells grok how to use its Resonant Lattice Memory. `AGENTS.md` loads after this file and wins.
> Copy this to your grok repo root as `AGENT.md`.

You have a **Resonant Lattice Memory (RLM)** available through grok's memory system.

- **Search before assuming.** When you encounter an unfamiliar file, symbol, concept, host, path,
  or prior decision, call `memory_search` on it BEFORE analyzing or acting. Prior sessions have
  likely grounded it - the lattice turns exploration into an informed audit rather than a guess.
- **Situational memory is auto-loaded** at session start and after compaction. Treat recalled
  facts as soft and fallible (approximate, possibly stale), not verbatim law.
- **Authority order:** your project kit (`AGENTS.md` and any decisions/session docs) and live
  checks win over recalled lattice facts on any conflict. Facts marked **[PRIORITY]** are
  user-pinned authority - weight them heavily and never act against them.
- **Writing memory:** to durably remember a fact or lock a decision, use your `rlm_remember` and
  `rlm_pin` tools (if the rlm-memory MCP server is installed). They write into the lattice
  (embedded, deduplicated; `rlm_pin` marks it `[PRIORITY]` authority) and persist across sessions.
  When you and the user lock a decision or a standing rule, `rlm_pin` it. Do NOT hand-edit grok's
  own memory files (a native `remember` / direct `MEMORY.md` edit is transient and gets overwritten
  by the lattice each session). The session ingest also captures whatever you clearly state.
- **Pruning and superseding memory:** you can also curate the lattice, not just add to it.
  `rlm_forget` prunes a fact (inverse of `rlm_remember`) and `rlm_unpin` drops a fact's `[PRIORITY]`
  authority while keeping the fact (inverse of `rlm_pin`). Target a fact by its `content` (as shown
  in your memory) or by its `id`; if the text is ambiguous you get candidate ids back and nothing is
  deleted, so a destructive edit is never a fuzzy guess. To **supersede** a stale fact, `rlm_forget`
  it and `rlm_remember` the corrected version. Edits hit the lattice immediately; your projected
  memory reflects them next session.
