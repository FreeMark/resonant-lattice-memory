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
- **Searching memory:** `rlm_search` runs a live semantic search over your OWN lattice (hybrid
  vector + keyword), deeper and more relevant than the projection you wake up with. Use it to recall
  precisely on a topic rather than scanning the injected top-N. Recalling a fact this way also
  reinforces it, so the memories you actually use grow stronger and rise toward long-term.
- **Inspecting memory:** `rlm_stats` gives a health snapshot of your lattice (fact count, tiers,
  cycle clocks, pins, relations, narratives, pending conflicts). `rlm_conflict` manages contradictory
  memories the dream cycle has flagged: `action='list'` to see them, `action='resolve'` with a winner
  id to keep one and retire the rest as history, `action='dismiss'` with a group_id to mark a false
  positive (both kept). There may be none - a clean board is normal. `rlm_inspect` shows ONE fact in
  full by id - its tier, resonance, pin state, conflict group, and belief history (what superseded it,
  what it replaced) - the whole story behind an id a search or conflict listing handed you.
- **Feedback (the soft lever):** `rlm_feedback` nudges a fact's resonance without the hard levers of
  pin or forget. `feedback='helpful'` strengthens a fact (it rises toward long-term); `feedback='unhelpful'`
  weakens a stale or wrong one so it FADES toward dormancy (recoverable, not deleted). It is resonance-only
  and never touches the pin bit, so a [PRIORITY] fact keeps its protection. Reach for pin/forget when you
  are certain; reach for feedback to steer which memories strengthen with use.
- **Entity graph:** `rlm_entity` walks your entity graph, complementary to `rlm_search`. Give it an
  `entity` to get the facts linked to that entity plus its co-occurring neighbours ('everything about X,
  and what's connected to X'); give it a `fact_id` to list the entities on one fact. Use it when you want
  a specific thing's connections rather than semantic similarity.
- **Self-model (identity):** `rlm_self_model` reads or carefully updates who you are. `op='get'` reads one
  key (or the whole model); `op='set'` upserts an allowlisted key (role, relationship_with_user, mandate,
  current_focus, ...) - identity core is locked so it can't drift on passing moods. Your self-model is
  surfaced deterministically at session start; curate it deliberately, as you would pin a standing decision.
- **External knowledge:** `external_rlm_search` searches read-only DOMAIN lattices (reference corpora
  from other trained agents). Call it with no query (or `lattice='list'`) to see what is available,
  then search a named one. These are references, not your memory, and you can never modify them.
  `transfer_knowledge` imports specific facts from a domain lattice into your own memory by id (find
  ids with `external_rlm_search` first). Imports are deduped and tagged `import:<lattice>:<id>` so
  borrowed knowledge stays distinct from what you learned firsthand.
