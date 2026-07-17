# RLM memory for the grok CLI agent

Give the [xAI `grok` CLI](https://github.com/xai-org) coding agent a **Resonant Lattice Memory
(RLM)**: grounded, consolidated, cross-session memory that the agent wakes up already knowing,
curates through tools, and can reason over as a typed knowledge graph.

This integration is **out-of-loop and uses grok's own seams** - no fork of grok, no plugin API.
RLM (running on a node with ollama) is the source of truth; grok's *native* memory engine is the
delivery layer. It is designed so that an agent (grok itself, or another coding agent) can read
this file and stand the whole thing up on a fresh grok install.

**Contents:** [Architecture](#architecture) · [Tool surface](#the-tool-surface) ·
[Memory scope](#memory-scope-per-repo-or-global) · [**Setup**](#setup) ·
[Config reference](#configuration-reference) · [Relation graph](#the-relation-graph) ·
[Per-turn prefetch](#per-turn-prefetch) · [What's implemented](#whats-implemented) ·
[Troubleshooting](#troubleshooting) · [Privacy](#privacy)

## Architecture

```
 grok CLI  (client machine)                        RLM node  (ollama + the RLM package)
 +----------------------------------+              +---------------------------------------+
 | PreCompact hook  (write path)    |   scp+ssh    | grok-agent-rlm/  (instance dir)       |
 |   snapshot chat_history.jsonl    |------------->|   resonant_lattice_memory.db  (truth) |
 |   -> detached worker -> node     |              |   rlm_ingest.py     consolidate       |
 |                                  |              |     -> facts + entities + RELATIONS   |
 | SessionStart hook  (read path)   |   ssh        |     -> dream cycle + narrative        |
 |   pull projection <--------------|<-------------|   rlm_export_memory.py   project      |
 |   write ~/.grok/memory/MEMORY.md |              |   rlm_*.py  (the 18 tool back-ends)   |
 |                                  |              |                                       |
 | UserPromptSubmit hook (prefetch) |   ssh        |   rlm_search.py --no-reinforce        |
 |   -> detached worker; recall on  |------------->|     -> <resonant_memory> block        |
 |   the user's message -> local    |              |                                       |
 |   block; rlm_prefetch serves it  |              |                                       |
 |                                  |              |                                       |
 | MCP server  (active tools)       |   ssh (18)   |   reason model  (extraction)          |
 |   rlm_mcp_server.py <------------|------------->|   relation model (triple slot-fill)   |
 |                                  |              |   embed model   (nomic-embed-text)    |
 | grok native memory engine        |  inject + memory_search                              |
 +----------------------------------+              +---------------------------------------+
        connection: ~/.grok/rlm-grok.conf  (RLM_DIR / RLM_PY; + SSH_HOST / SSH_KEY for a remote node)
```

**Transport is local by default.** When grok runs on the **same machine** as the RLM instance (the
common case, and simplest), the hooks run the node scripts **directly** - no ssh, no key. Only when
grok is on a **different** machine than the node do you set `SSH_HOST` (+ `SSH_KEY`) and the hooks
switch to ssh/scp. Everything below works identically either way; the `scp+ssh` / `ssh` arrows in the
diagram are direct local calls when co-located.

Four planes, one lattice:

1. **Passive write (PreCompact):** compacting a session snapshots the transcript *before*
   compaction collapses it, ships it to the node, and RLM consolidates it - extracting grounded,
   resonance-ranked facts, entities, and typed relations, then running one dream cycle (decay /
   tier-promotion / conflict-detect) and a rolling narrative built from the *whole* session's born
   facts (a hierarchical digest, so a long multi-window compact is narrated in full, not just its
   tail). grok does **not** fire `SessionEnd`
   on a normal exit, so the convention is: **compact before you exit**.
2. **Passive read (SessionStart):** a hook pulls a Markdown projection of the lattice into grok's
   native `MEMORY.md`. grok's engine indexes it and injects it first-turn and answers
   `memory_search` on demand. The projection leads with the self-model, hoists pinned
   facts into an **authority** block, quarantines unresolved conflicts into a **contested** block,
   and ends with the recent narrative. Its sections are packed to grok's chunk window (see
   [Per-turn prefetch](#per-turn-prefetch) below) so the indexer injects whole topics, not fragments.
3. **Per-turn prefetch (UserPromptSubmit):** grok's engine only auto-injects memory on the *first*
   turn and after compaction; every turn in between is pull-only. This hook closes that gap. On each
   submitted prompt it detaches a worker that recalls against the user's **actual message** and
   writes a `<resonant_memory>` block to a local file; the `rlm_prefetch` MCP tool serves that block
   instantly (no node round-trip). It gives grok hermes-style per-turn recall for the cost of one
   argument-free tool call. See [Per-turn prefetch](#per-turn-prefetch).
4. **Active tools (MCP):** a small stdio MCP server exposes the lattice as 18 tools so the agent
   can operate its memory mid-session, not just wake up with a projection.

RLM is the **sole writer** of grok's memory (grok's own auto-save / dream / compaction-flush are
turned off), so the store stays a clean projection of the lattice. All temporal dynamics are driven
by **memory cycles, not wall-clock**, and extraction is anti-fabrication (verbatim source quotes;
drop rather than guess).

## The tool surface

The MCP server (`hooks/rlm_mcp_server.py`) exposes **18 tools**. Each SSHes to a node script that
runs against the agent's own lattice.

| Group | Tool | What it does |
|-------|------|--------------|
| **write / curate** | `rlm_remember` | durable write (embedded, deduped, provenance-tagged) |
| | `rlm_pin` | write **and** pin as `[PRIORITY]` authority (never forgotten, surfaced first) |
| | `rlm_forget` | prune a fact (by id or exact content; ambiguous text returns candidates, never fuzzy-deletes) |
| | `rlm_unpin` | drop `[PRIORITY]` but keep the fact |
| **recall** | `rlm_prefetch` | serve the per-turn `<resonant_memory>` block precomputed for the current message (instant, no args); reinforces on read |
| | `rlm_search` | hybrid vector + keyword search over the OWN lattice; a recall reinforces the fact |
| | `external_rlm_search` | read-only search of DOMAIN lattices (reference corpora dropped in `lattices/`) |
| | `transfer_knowledge` | import specific facts from a domain lattice by id (deduped, tagged `import:<lattice>:<id>`) |
| **feedback** | `rlm_feedback` | soft resonance nudge (`helpful` / `unhelpful`) - steer strength without pin/forget |
| **inspect** | `rlm_inspect` | one fact in full + its belief history (what superseded it / what it replaced) |
| | `rlm_entity` | walk the entity graph (facts about an entity + co-occurring neighbours, or entities on a fact) |
| **relations** | `rlm_relational` | typed, directed graph query (`what runs on the node`, `how is X connected`) |
| | `rlm_infer` | bounded multi-hop inference with the supporting path + per-hop-decayed confidence (a hypothesis) |
| **identity** | `rlm_self_model` | read, or update an allowlisted identity key (role / mandate / current_focus …) |
| **health** | `rlm_stats` | fact count, tiers, pins, cycle clocks, entities, relations, narratives, pending conflicts |
| | `rlm_dream` | dream-cycle health: tier flow + promotion-ready counts, dwell maturity, decay/fading, contested facts, abstraction/gist output, dials in effect |
| | `rlm_narrative` | recent session arcs (throughline / decisions / open loops / closed), newest first - read past the projected top-N |
| | `rlm_conflict` | list / resolve / dismiss contradictions the dream cycle flagged |

The agent's operating rule for these lives in [`rules/AGENT.md`](rules/AGENT.md) (search-on-encounter,
authority order, when to pin vs feedback vs forget).

## Memory scope: per-repo or global

There is **one lattice** (the node brain); the SessionStart hook projects it into grok's memory.
Where it lands depends on `RLM_MEMORY_SCOPE` (set in the hook JSON's `env`):

- **`workspace`** (default): written into the **per-repo** grok memory. grok keys workspace memory
  by the git `origin` remote (shared across all clones/worktrees of the same repo; a non-git folder
  is keyed by its path). Each repo must be **bootstrapped once** (`bootstrap.sh`). *One brain,
  surfaced in the repos you bootstrap.*
- **`global`**: written into grok's **global** memory (`~/.grok/memory/MEMORY.md`), searched in
  **every** session of **every** repo. No per-repo bootstrap. Use when this agent is primarily your
  RLM operator and you want the lattice everywhere.

---

# Setup

Follow the parts in order. Each ends with a **verify** step - do not proceed until it passes.
Replace `NODE`, `KEYPATH`, etc. with your values. Throughout, the **node** is the Linux box running
ollama + the RLM package; the **client** is the machine running grok (may be the same box).

## Prerequisites

- A **node** (the machine that runs the RLM instance) with **ollama** and three models pulled. This
  can be the **same machine grok runs on** (local, the default) or a separate box (remote, over SSH).
  - a **reason model** (extraction/consolidation) - favour quality; a mid-size local or cloud model.
  - a **relation model** (triple slot-filling) - a **small LOCAL** model (granite-class) is ideal
    and cheap; it runs once per fact/window.
  - an **embed model** - `nomic-embed-text`.
- The **RLM package** (this repo's `resonant_lattice/`) on the node, importable by a Python that has
  its deps (`sqlite_vec`, `numpy`, `pyyaml`, and the `agent` MemoryProvider ABC). A venv is fine.
- **grok CLI** on the client machine (`grok --version`).
- **Only for a remote node:** an **SSH private key** on the client that logs in to the node
  (`ssh -i KEYPATH user@NODE true`). Not needed when grok and the node are the same machine.

## Part 1 - Node (the brain)

1. **Put the RLM package on the node** (default location `~/he-rlm/`, so `~/he-rlm/resonant_lattice/`
   exists). Override the location with `RLM_PACKAGE_DIR` if elsewhere.
2. **Create the instance dir and copy the node scripts:**
   ```bash
   ssh user@NODE 'mkdir -p ~/grok-agent-rlm/{incoming,lattices}'
   scp integrations/grok/node/*.py user@NODE:~/grok-agent-rlm/
   ```
   `incoming/` receives transcript snapshots; `lattices/` holds optional read-only domain lattices
   (`<domain>.db`) for `external_rlm_search`.
3. **Create `config.yaml`** in the instance dir from
   [`node/config.example.yaml`](node/config.example.yaml). Set the three model names + endpoints,
   and (recommended) enable the relation graph. The example file is fully commented; the sections
   that matter are summarized in [Configuration reference](#configuration-reference) below.
4. **Verify the node** (uses the RLM-package Python - the same one you will name as `REMOTE_PY`):
   ```bash
   ssh user@NODE 'cd ~/grok-agent-rlm && REMOTE_PY rlm_stats.py'
   # expect JSON: {"ok": true, "total_facts": 0, ...}  (0 facts on a fresh lattice)
   ```
   If it prints stats JSON, the package imports, the config loads, and the db opens.

## Part 2 - Client (the grok machine)

1. **Enable grok memory with RLM as sole writer** - merge
   [`grok-memory-config.toml`](grok-memory-config.toml) into `~/.grok/config.toml`. This turns grok's
   memory engine ON (injection + `memory_search`) and grok's own writers OFF (so the store stays a
   clean RLM projection).
2. **Connection file** - copy [`hooks/rlm-grok.conf.example`](hooks/rlm-grok.conf.example) to
   `~/.grok/rlm-grok.conf`. **Local (default, grok on the RLM machine)** - just the instance dir and
   its Python, no ssh:
   ```
   RLM_DIR=/home/user/grok-agent-rlm
   RLM_PY=/home/user/venv/bin/python3   # the Python that imports the RLM package
   ```
   **Remote node (over SSH)** - additionally set the host and key; then `RLM_DIR`/`RLM_PY` are paths
   *on the node*:
   ```
   SSH_HOST=user@NODE            # setting this switches the transport to ssh/scp
   SSH_KEY=~/.ssh/your_key
   ```
   Transport is chosen automatically: **no `SSH_HOST` = local**, `SSH_HOST` set = ssh. Every hook and
   the MCP server read this one file, so switching transport is a one-line change and an update is a
   plain copy of the published file. (`REMOTE_DIR`/`REMOTE_PY` are accepted as aliases for
   `RLM_DIR`/`RLM_PY` for older confs.)
3. **Hooks** - copy the config loader + hook scripts + the three hook JSONs into `~/.grok/hooks/`:
   ```bash
   cp integrations/grok/hooks/rlm_grok_conf.py        ~/.grok/hooks/
   cp integrations/grok/hooks/rlm_precompact_dispatch.py ~/.grok/hooks/
   cp integrations/grok/hooks/rlm_ingest_worker.py    ~/.grok/hooks/
   cp integrations/grok/hooks/rlm_sessionstart_memory.py ~/.grok/hooks/
   cp integrations/grok/hooks/rlm_prefetch_dispatch.py ~/.grok/hooks/  # per-turn prefetch
   cp integrations/grok/hooks/rlm_prefetch_worker.py  ~/.grok/hooks/
   cp integrations/grok/hooks/rlm-ingest.json         ~/.grok/hooks/   # PreCompact -> dispatch
   cp integrations/grok/hooks/rlm-memory.json         ~/.grok/hooks/   # SessionStart -> project
   cp integrations/grok/hooks/rlm-prefetch.json       ~/.grok/hooks/   # UserPromptSubmit -> prefetch
   ```
   **Edit the `command` in each `*.json`** to your Python and the absolute path to the hook script
   (they are launched by grok, so use an absolute interpreter path, e.g. the system python). For
   **global scope**, set `"env": { "RLM_MEMORY_SCOPE": "global" }` in `rlm-memory.json`. The prefetch
   hook needs no env; it keys its block file by the workspace path so concurrent repos don't collide.
4. **Active MCP tools** - copy the server and register it:
   ```bash
   cp integrations/grok/hooks/rlm_mcp_server.py ~/.grok/hooks/
   ```
   Merge [`grok-mcp-config.toml`](grok-mcp-config.toml) into `~/.grok/config.toml`, editing
   `command` (your python) and the `args` path (absolute path to `rlm_mcp_server.py`). The server
   reads the **same `~/.grok/rlm-grok.conf`** for its connection.
5. **Operating rule** - copy [`rules/AGENT.md`](rules/AGENT.md) to your grok repo root. (It is *not*
   your `AGENTS.md`; `AGENTS.md` loads after it and wins.)
6. **Verify the client:**
   ```bash
   # (a) connection resolves
   cd ~/.grok/hooks && python -c "import rlm_grok_conf as C; print('configured:', C.configured())"
   # -> configured: True

   # (b) MCP server lists 18 tools and reaches the node
   printf '%s\n%s\n' \
     '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
     '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | python ~/.grok/hooks/rlm_mcp_server.py
   # -> a result listing 18 tools (rlm_pin ... rlm_infer)

   # (c) per-turn prefetch: feed a fake prompt through the hook, then serve the block
   echo '{"sessionId":"t","cwd":"'$PWD'","workspaceRoot":"'$PWD'","prompt":"what do we know about the node setup"}' \
     | python ~/.grok/hooks/rlm_prefetch_dispatch.py && sleep 3
   printf '%s\n%s\n' \
     '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
     '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"rlm_prefetch","arguments":{}}}' \
     | python ~/.grok/hooks/rlm_mcp_server.py
   # -> a <resonant_memory> block "precomputed Ns ago for the current message"
   ```

## Part 3 - Scope + bootstrap

- **Global scope** (simplest): with `RLM_MEMORY_SCOPE=global` in `rlm-memory.json`, no per-repo
  bootstrap is needed - the projection is written to `~/.grok/memory/MEMORY.md` for every session.
- **Workspace scope:** run [`bootstrap.sh`](bootstrap.sh) once **from inside each repo** you want the
  lattice in. It forces grok to create the per-repo memory dir (which the SessionStart hook then
  populates). Requires grok on PATH and `[memory]` enabled.

## Part 4 - Verify end to end

1. **Read path:** run the SessionStart hook by hand (global scope shown) and confirm it writes a
   projection from the node:
   ```bash
   RLM_MEMORY_SCOPE=global python ~/.grok/hooks/rlm_sessionstart_memory.py < /dev/null
   head -5 ~/.grok/memory/MEMORY.md    # -> the projection header + sections
   ```
2. **Active write:** in a grok session, ask it to `rlm_remember` a fact, then `rlm_search` for it.
   Or drive the MCP server directly (see Part 2 verify) with a `tools/call` for `rlm_stats`.
3. **Write path:** run a real session, then **compact** it. The PreCompact hook snapshots the
   transcript and detaches a worker; watch the node ingest (`~/grok-agent-rlm/ingest.log`) and the
   fact count climb (`rlm_stats.py`). On your next session, the new facts are in the projection.

---

# Configuration reference

The instance `config.yaml` is flat RLM-plugin keys (see [`node/config.example.yaml`](node/config.example.yaml)
for every key, commented). The load-bearing sections:

**Models + connection**
```yaml
reason_model: <your-reason-model>          # extraction/consolidation
relation_model: <small-local-model>        # triple slot-filling (per fact/window; keep it local)
embed_model: nomic-embed-text:latest
ollama_endpoint_reason:   http://YOUR_OLLAMA_HOST:11434
ollama_endpoint_relation: http://YOUR_OLLAMA_HOST:11434
ollama_endpoint_embed:    http://YOUR_OLLAMA_HOST:11434
memory_reason_max_concurrency: 1           # serialize reason/LLM calls into one lane
```

**Retention + passive promotion** - because grok's read path is a projection that does not
reinforce RLM resonance, set `initial_resonance` **above** `promotion_resonance_threshold` so a
durable fact can promote to the long tier before decaying out.
```yaml
initial_resonance: 6
promotion_resonance_threshold: 4
decay_per_cycle: 0.5
reinforce_on_recall: true                  # a fact the agent rlm_search-es strengthens (use drives promotion)
recall_bump: 2.0
```

**Roadmap layers** (each costs model calls in/after the dream): `gist_before_prune`,
`enable_self_model` (+ a `self_model_seed`), `enable_narrative` (a rolling autobiographical
paragraph; point `narrative_model` at a small local model).

**Feedback + self-model** (read by the node tool scripts):
```yaml
feedback_helpful_delta: 1.0                # rlm_feedback helpful (gentle, < recall_bump)
feedback_unhelpful_delta: -3.0             # unhelpful fades a wrong/stale fact faster
self_model_writable_keys: [role, relationship_with_user, mandate, current_focus, values, communication_style]
```

**The relation graph** - see the next section.

# The relation graph

Off by default; turn it on to give the agent a **typed, traversable map of how its system is wired**
(`rlm_relational` / `rlm_infer`). The design: a *closed* relation vocabulary makes relations recur
into a queryable graph, and node canonicalization makes multi-hop chains form.

```yaml
enable_relations: true
relation_extract_llm: true                 # small local model slot-fills triples per new fact
relation_min_confidence: 0.5
relation_require_entity: false             # strict binding (drops fragment noise at a recall cost)
relation_extract_from_transcript: true     # ALSO mine the raw ingest window for dependency/decision edges
relation_vocabulary:                        # a CLOSED set matched to your DOMAIN
  [runs_on, serves, uses, set_to, produces, supersedes, depends_on, part_of]
relation_examples: |                        # domain few-shots (include one mapping to [] for a non-relational note)
  Examples:
  Note: "..." -> [{"subject":"...","relation":"serves","object":"..."}]
  Note: "an introspective sentence" -> []
entity_vocabulary:                          # high-precision allowlist of DOMAIN terms the patterns miss
  [rlm_pin, initial_resonance, relational, resonance, lattice, dream cycle]   # -> become graph nodes
entity_aliases:                             # unify surface forms into ONE node (per thing)
  "the node": "your-node"
  "toolA__toolB": "toolB"                   # e.g. MCP-namespaced tool ids -> the tool
```

**Presets** for `personal` / `operational` / `technical` domains are documented in `config_schema.py`
and the [configurator page](https://freemark.github.io/resonant-lattice-memory/). Empty
`relation_vocabulary` = legacy free-form behaviour. The graph is a **fallible secondary substrate**
(roughly two-thirds precise from a small local model on dense prose) - grounded, not gospel; it grows
and improves as the lattice does. A diagnostic, `node/rlm_triple_diag.py`, measures where triples die
in your own corpus.

# Per-turn prefetch

grok's native engine auto-injects memory on the **first turn** and after **compaction** - and
nowhere else. Every turn in between, memory is *pull-only*: it surfaces only if grok volunteers a
`memory_search`. A reference in-loop agent (hermes) instead recalls on **every** turn, so the right
memories arrive as a lens on what the user just said, whether or not the model thought to ask. This
integration brings grok as close to that as its out-of-loop seams allow, in two moves:

**1. The prefetch bridge (per-turn recall).** grok ignores passive-hook stdout, so a hook cannot
inject context directly - but it *can* have side effects. On every submitted prompt, the
`UserPromptSubmit` hook (`rlm_prefetch_dispatch.py`) detaches a worker that recalls against the
user's **actual message** (`rlm_search.py --no-reinforce`, ~0.5-3s, no LLM) and writes a
`<resonant_memory>` block to `~/.grok/rlm-queue/prefetch/<workspace-hash>.md`. The `rlm_prefetch`
MCP tool then serves that block **instantly** (a local file read, no node round-trip) with no query
for the model to formulate - its biggest failure mode. Reading the block reinforces the recalled
facts (`rlm_reinforce.py`), so the split is honest: **precompute does not reinforce, use does**.
Guards mirror the hermes prefetch: a trivial-ack skip (`yes`, `continue`), a same-topic overlap gate
that reuses a fresh block instead of recomputing, and a cache-miss fallback to a live recall on the
captured prompt. The operating rule in [`rules/AGENT.md`](rules/AGENT.md) tells grok to call
`rlm_prefetch` **first** on any turn that touches prior work.

*Why a tool call at all?* No grok seam can force context into the model mid-session (only
`PreToolUse` may return data, and only to allow/deny a call). So the block is *staged* by the hook
and *served* by one argument-free, instant tool call - as close to autonomous per-turn recall as the
wrapper permits.

**2. Chunk-aligned projection.** grok's indexer chunks memory files at ~1600 chars and ranks
*chunks* for both its native injections and `memory_search`. A long category emitted as one list gets
split mid-list into headingless fragments that rank poorly. `rlm_export_memory.py` now packs every
section to <=1400 chars under its own heading (continuations get `(cont.)`), so each indexed chunk is
a coherent, self-contained topic - what grok injects and finds is whole topics, not fragments.

# What's implemented

The integration, end to end:

- **Passive loop** - PreCompact snapshot (survives compaction) → detached worker → node ingest;
  ACP (grok session-update) transcript parser; per-window consolidation → one post-ingest dream
  cycle → rolling narrative; SessionStart projection with a self-model header, a pinned
  **authority** block, a **contested** (unresolved-conflict) quarantine, and the recent narrative.
  UTF-8-clean write path. The worker launches the node ingest with `python -u`, so per-window
  progress lines stream live instead of flushing only at exit.
- **Ingest observability** - `rlm_watch_ingest.py` (with the `rlm-watch.cmd` launcher) watches a
  `/compact` -> ingest "memory cycle" to completion and signals when it is safe to continue the
  conversation. Buffer-immune: it keys off the `rlm_ingest.py` process plus the live `semantic_facts`
  count, not the block-buffered log.
- **Re-compact dedup** - grok's `chat_history.jsonl` is append-only, so compacting the same session
  twice re-snapshots from turn 1. The ingest keeps a per-session line high-water mark
  (`ingest_watermarks`, keyed on the stable grok session id) and mines only the new tail on a later
  compact, so re-compacts stay cheap and the lattice does not accrue near-duplicate facts. The mark
  advances only on a clean finish, so a crash safely re-processes.
- **Per-turn prefetch** - a `UserPromptSubmit` hook precomputes a `<resonant_memory>` block for the
  current message; the `rlm_prefetch` tool serves it instantly and reinforces on read. Plus a
  chunk-aligned projection so grok's native first-turn / post-compaction injection surfaces whole
  topics, not fragments. Closes the last hermes<->grok recall gap the wrapper allows.
- **18 MCP tools** - write/curate (`rlm_remember`, `rlm_pin`, `rlm_forget`, `rlm_unpin`), recall
  (`rlm_prefetch` per-turn block; `rlm_search` with recall-reinforcement; `external_rlm_search` +
  `transfer_knowledge` for domain lattices), feedback (`rlm_feedback`), inspect (`rlm_inspect`,
  `rlm_entity`), relations (`rlm_relational`, `rlm_infer`), identity (`rlm_self_model`,
  allowlist-gated), health (`rlm_stats`, `rlm_conflict`).
- **Domain-configurable relation graph** - closed `relation_vocabulary` + constrained slot-filling
  (relations recur), `entity_aliases` node canonicalization (chains form), `entity_vocabulary`
  (domain tool/config/concept names become graph nodes and survive strict binding), and
  `relation_extract_from_transcript` (dependency/decision edges mined from the raw window, merged
  with per-fact relations). Core, domain-agnostic, backward-compatible.
- **Recall reinforcement + curation** - `rlm_search` strengthens what it recalls; `rlm_feedback` is
  the soft lever between pin/forget; `initial_resonance > promotion_threshold` lets durable facts
  passively promote under a read-only projection.
- **Local-or-remote transport** - the hooks + MCP server run the node scripts **directly** when grok
  is on the RLM machine (default, no ssh), or over ssh/scp when `SSH_HOST` is set. Transport is
  centralized in `rlm_grok_conf.py` (`run` / `push`), chosen by config, so nothing is hardcoded.
- **Transform-free deploy** - every hook and the MCP server take their connection from one
  `~/.grok/rlm-grok.conf`, so updating any piece is a plain copy of the published file.

# Troubleshooting

- **`configured: False`** - local mode needs `RLM_DIR` + `RLM_PY`; remote (ssh) mode additionally
  needs `SSH_HOST` + `SSH_KEY`. Check the file is at `~/.grok/rlm-grok.conf`.
- **Local mode: `sh: not found` / commands do nothing** - local transport uses `sh -c`, so it targets
  a POSIX host (the normal case when grok runs on the Linux RLM node). On a Windows-only box, run the
  node over SSH instead, or provide a POSIX `sh` (e.g. Git Bash) on PATH.
- **MCP server errors / no tools** - run it by hand (Part 2 verify); a Python import error means the
  server can't find `rlm_grok_conf.py` (it must sit beside it in `~/.grok/hooks/`).
- **`ModuleNotFoundError` on the node** - `REMOTE_PY` is not the Python that has the RLM package +
  deps. Point it at the venv/interpreter that can `import resonant_lattice` (or set `RLM_PACKAGE_DIR`).
- **Projection empty** - the lattice has no non-superseded facts yet (fresh install), or (workspace
  scope) the repo was not bootstrapped. Check `rlm_stats.py` and `bootstrap.sh`.
- **No relations appearing** - `enable_relations` + `relation_extract_llm` must be on and a
  `relation_vocabulary` set; run `node/rlm_triple_diag.py` to see where triples are dropped.
- **`rlm_prefetch` says "not primed"** - the `UserPromptSubmit` hook (`rlm-prefetch.json`) is not
  installed or has not fired yet; check `~/.grok/rlm-queue/prefetch.log` and that a `<hash>.md` block
  exists under `~/.grok/rlm-queue/prefetch/`. Until then, `rlm_prefetch` falls back to a live recall
  on the captured prompt; `rlm_search` with an explicit query always works.
- **`external_rlm_search lattice=list` returns "no lattices" (or `rlm_stats`/`rlm_conflict` stall)
  inside a live grok session** - fixed in v1.6.6. Older `rlm_grok_conf.py` let an input-less node
  call inherit the server's stdin (grok's JSON-RPC pipe), which deadlocked to a timeout under the
  v1.6.4 concurrent dispatch. Update `rlm_grok_conf.py` to the current version (a plain copy) and
  restart the grok session. Named search was never affected.
- **"Is the memory cycle done? Can I continue after `/compact`?"** - run `rlm-watch.cmd` (or
  `python ~/.grok/hooks/rlm_watch_ingest.py`). It refreshes until the ingest finishes, then rings the
  bell and prints a "safe to continue" banner. It watches the `rlm_ingest.py` process + the live
  `semantic_facts` count, so it stays correct even though the ingest log is block-buffered (per-window
  lines can lag). `--once` prints a single snapshot instead of looping.
- **grok wrote to `MEMORY.md` itself** - grok's own writers are still on; re-check
  `grok-memory-config.toml` is merged (RLM must be the sole writer).

# Privacy

grok's native memory index embeds `MEMORY.md` via xAI's embedding API, so projected memory content
leaves your machine the same way your session already does when you use grok. Connection secrets
(SSH key path, node host) live only in `~/.grok/rlm-grok.conf` on your machine, never in the repo.
If you need fully local memory, the RLM package supports local-only retrieval directly (see the main
repo); this integration deliberately trades that for grok's native prefetch + tool UX.
