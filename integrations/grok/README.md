# RLM memory for the grok CLI agent

Give the [xAI `grok` CLI](https://github.com/xai-org) coding agent a **Resonant Lattice Memory**:
grounded, consolidated, cross-session memory that the agent wakes up already knowing, and can
search on demand.

This integration is **out-of-loop and uses grok's own seams**: no fork of grok, no plugin API.
RLM is the source of truth; grok's *native* memory engine is the delivery layer.

## How it works

```
 grok CLI (your machine)                         RLM node (ollama + the RLM package)
 +-------------------------------+               +------------------------------------+
 | PreCompact hook               |               | grok-agent-rlm/ instance           |
 |   snapshot chat_history.jsonl |   ssh/scp     |   lattice.db  (source of truth)    |
 |   -> ship to node ------------+-------------->|   rlm_ingest.py  (consolidate)     |
 |                               |               |   rlm_export_memory.py (project)   |
 | SessionStart hook             |               |   rlm_pin_facts.py  (pin authority)|
 |   pull lattice projection ----+<--------------+   your reason/relation/embed models|
 |   write ~/.grok/memory/.../   |               +------------------------------------+
 |        MEMORY.md              |
 | grok native memory engine     |   first-turn injection (prefetch) + memory_search (postfetch)
 +-------------------------------+
```

- **Write path (PreCompact):** when you compact a session, a hook snapshots the transcript
  *before* compaction collapses it, ships it to the node, and RLM consolidates it into the
  lattice (grounded, resonance-ranked, anti-fabrication).
- **Read path (SessionStart):** a hook pulls a Markdown projection of the lattice and writes it
  into grok's native workspace `MEMORY.md`. Grok's engine indexes it and does the rest:
  - **first-turn injection**: situational memory auto-loaded at session start + after compaction (prefetch);
  - **`memory_search` tool**: the agent searches memory on demand for deep recall (postfetch), as a visible tool call.

RLM is the **sole writer** of grok's memory (grok's own auto-save/dream/flush are turned off), so
the store stays a clean projection of the lattice rather than flat session summaries.

> Trigger note: `PreCompact` is used because grok does **not** fire `SessionEnd` on a normal
> interactive exit. The convention is to run a compaction before exiting a session.

## Layout

| Path | Runs on | What |
|------|---------|------|
| `node/` | the RLM node | the RLM instance: consolidate ingest, project to Markdown, pin authority facts |
| `hooks/` | the machine running grok | grok hooks (JSON) + their scripts, parameterized via a small conf file |
| `rules/AGENT.md` | grok repo root | the memory operating rule grok follows (search-on-encounter, authority order) |
| `grok-memory-config.toml` | `~/.grok/config.toml` | the `[memory]` snippet: enable grok memory, RLM as sole writer |
| `bootstrap.sh` | the machine running grok | one-time per repo: make grok create its memory dir |

## Setup

Assumes you already have the RLM package on a node (or locally) with ollama and an RLM instance
dir (a `config.yaml` + a `lattice.db`). See the main repo README for the RLM package itself.

1. **Node:** copy `node/*` into your RLM instance dir (e.g. `~/grok-agent-rlm/`) and fill in
   `config.yaml` from `node/config.example.yaml` (your reason/relation/embed model endpoints).
   Ensure the RLM package is importable (default expects it at `~/he-rlm`; override with
   `RLM_PACKAGE_DIR`).
2. **grok config:** merge `grok-memory-config.toml` into `~/.grok/config.toml`.
3. **Connection:** copy `hooks/rlm-grok.conf.example` to `~/.grok/rlm-grok.conf` and set your
   SSH key, node host, remote instance dir, and remote python.
4. **Hooks:** copy `hooks/rlm_*.py` into `~/.grok/hooks/`, and the two `*.json` hooks into
   `~/.grok/hooks/` (edit the `command` paths in each JSON to your python + hook-script paths).
5. **Rule:** copy `rules/AGENT.md` to your grok repo root (it is not your `AGENTS.md`).
6. **Bootstrap (per repo):** run `bootstrap.sh` from inside the repo once, so grok creates its
   memory dir.

Then just use grok: compact before exiting (ingest), and each new session wakes up with the
lattice (prefetch). `memory_search` is available for deep recall.

## Privacy note

grok's native memory index embeds `MEMORY.md` via xAI's embedding API, so memory content leaves
your machine the same way your session already does when you use grok. If you need fully local
memory, the RLM package supports local-only retrieval directly (see the main repo); this
integration deliberately trades that for grok's native prefetch + tool UX.
