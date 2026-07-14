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
- **You are not the memory writer.** The lattice is written by the RLM pipeline (sole writer)
  from your sessions. State durable facts clearly so the next ingest captures them; do not
  hand-edit memory files.
