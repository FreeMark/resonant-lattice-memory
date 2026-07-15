# Resonant Lattice Memory

A **neuroplastic, local-first long-term memory** for AI agents. Facts behave like resonant
circuits: a memory that keeps getting struck stays in tune and rings loud (reinforced, promoted
to a durable tier); the ones nothing resonates with fade to silence (decay, prune). The result is
a memory that **keeps what matters and forgets the noise, on its own, driven by usage cycles, not
wall-clock timers.**

It ships as a drop-in plugin for [hermes-agent](https://hermes-agent.nousresearch.com), an
out-of-loop integration for the [xAI `grok` CLI agent](integrations/grok/), and a standalone
Python library (the store, recall, and consolidation run without any agent framework). It is
designed so **an agent can read this file, verify the system on its own machine, deploy it, and
configure it**, the same loop a human and agent used to build and harden it.

> **Why it matters:** an agent is only as coherent as its memory. Across compression, fresh
> sessions, and long horizons it has to surface the right thing, stay grounded (never fabricate),
> hold the rules it was given, and resist poisoning and staleness, all while bounding itself on
> its own. Each of those is a tested property here, and there are more (the `results/` files are
> the evidence). The aim is to make even small, local models coherent over the long haul, not
> only large chat models.

**Tune it visually:** the [**RLM Configurator**](https://freemark.github.io/resonant-lattice-memory/)
is a live, dependency-free page generated from `config_schema.py`. Every knob, its default, and
ready-made preset profiles (Scholar, Sovereign Vault, Overnight, Low-VRAM, and more) in one place.

---

## What's new

- **v1.5.0 (2026-07-15): the grok memory lifecycle is complete.** The out-of-loop grok integration
  now runs the whole loop, not just read and write:
  - **Curation.** Two inverse write tools: `rlm_unpin` (drop a fact's `[PRIORITY]` authority while
    keeping the fact, the inverse of `rlm_pin`) and `rlm_forget` (prune a fact, the inverse of
    `rlm_remember`). Together they let the agent supersede and curate its own memory, not just
    accumulate it. A fact is targeted by its content or its id; an ambiguous match returns candidate
    ids without acting, so a destructive edit is never a fuzzy guess.
  - **Dreaming.** The ingest driver now runs one full dream cycle after each ingest (per-window
    consolidation stays fast, then a single post-ingest dream), so the temporal dynamics (decay,
    tier promotion, prune, conflict detection) actually fire for a grok-only lattice. Pinned
    authority facts stay exempt from decay and prune.
  - **Narrative.** A rolling autobiographical paragraph is written per ingest so the agent wakes up
    with continuity. It runs on a configurable model (`narrative_model` / `narrative_endpoint`), so
    summarization can be offloaded to a small local model while the reasoning lane stays free.
  - **Unicode write fix.** Writes are sent as explicit UTF-8, so em-dashes and other non-ASCII
    characters survive the Windows-to-node hop instead of failing.

  The config template also gains the retention and layer dials, including a note that a read-only
  recall path needs `initial_resonance` above the promotion threshold for facts to passively reach
  the long tier.
- **v1.4.8 (2026-07-15): open-ended extraction categories.** The grok integration's config
  template (`integrations/grok/node/config.example.yaml`) replaces the fixed category enum in the
  extraction prompt with an open-ended rule: pick a self-evident, plain-language category, let
  natural overlap happen, and never drop or distort a fact just because it does not match a fixed
  label. The category is a hint for recall, not a gate on what gets remembered.
- **v1.4.7 (2026-07-14): grok memory scope (per-repo or global).** The grok integration's
  SessionStart projection now takes an `RLM_MEMORY_SCOPE` lever: `workspace` (default, per-repo
  memory keyed by git origin) or `global` (project the lattice into grok's global memory so it
  surfaces in every repo, written as a preserved managed block). The integration README gains a
  "Memory scope" section explaining the origin-keyed model and the trade-offs.
- **v1.4.6 (2026-07-14): grok can write to the lattice.** The grok integration gains durable
  memory-write tools, `rlm_pin` (write + pin as `[PRIORITY]` authority) and `rlm_remember` (write a
  durable fact), through a small dependency-free MCP server that routes the agent's writes into the
  node lattice (embedded, deduped, provenance-tagged). It is the write-side complement to
  `memory_search`, and the durable alternative to grok's native `remember`, which only touches the
  local file and is overwritten each session.
- **v1.4.5 (2026-07-14): memory for the grok CLI agent.** A new out-of-loop integration
  ([`integrations/grok/`](integrations/grok/)) gives the xAI `grok` coding agent an RLM memory
  through grok's own seams: a `PreCompact` hook ingests each session into the lattice, and a
  `SessionStart` hook projects the lattice into grok's native memory, so the agent wakes up
  already knowing it (first-turn injection) and can search it on demand (`memory_search`), with
  RLM as the sole writer. No fork of grok, no plugin API.
- **v1.4.4 (2026-07-14): operator-hardening.** From an expert-operator review of the config
  surface: real bug fixes (`detect_policy_conflicts` / `detect_procedural_conflicts` now actually
  thread from the provider into the store instead of being inert; `infer` hop-floor corrected so
  `hops=1` means no multi-hop; the six reason-model prompts promoted to first-class config dials
  sharing one source of truth), plus a structural `<authority_rules>` block that lifts pinned
  `[PRIORITY RULE]`s into their own section of the recall injection. Unit suite **140/0**.
- **v1.4.2 (2026-07-12): GPU blind-recall accelerator seam (off by default).** An optional GPU
  backend for the homomorphic blind-recall scoring: bit-exact parity with the CPU path,
  order-of-magnitude faster scoring, and fully inert unless enabled.
- **v1.4.1 (2026-07-11): bounded-RAM blind recall.** Homomorphic recall streams ciphertexts in
  auto-sized batches instead of loading the whole encrypted corpus at once. On a 4,952-fact
  corpus the recall working set dropped **5.0 GB to 0.65 GB** (7.7x) with **identical** top-10
  results.
- **v1.4.0 (2026-07-11): trained in the open, sealed after the fact.** A lattice trained entirely
  in plaintext was run through the full encryption stack afterward and sealed, and stayed
  searchable while encrypted. Added consolidation-debt retries and the `[SYNTHESIZED]` provenance
  label for facts the agent forms by reflecting on its own memory.

Full history in the [Releases](https://github.com/FreeMark/resonant-lattice-memory/releases).

---

## What's in this repo

| Path | What it is |
|---|---|
| `resonant_lattice/` | The plugin (runtime code, `plugin.yaml`, `recommended_config.yaml`, architecture docs, the **140-test** unit suite, the eval harness). |
| `integrations/grok/` | RLM memory for the **xAI `grok` CLI agent**, out-of-loop through grok's own hooks + native memory. See [its README](integrations/grok/README.md). |
| `tests/` | The behaviour / trust / scale / durability suite, plus the live end-to-end exercise `live_e2e.py`. |
| `results/` | All test evidence: per-test outputs, metrics (`.jsonl`), model-comparison summaries, and the single-file [`CONSOLIDATED_RESULTS.md`](results/CONSOLIDATED_RESULTS.md). |
| `docs/` | The [RLM Configurator](https://freemark.github.io/resonant-lattice-memory/) page (GitHub Pages), generated from `config_schema.py`. |
| `tools/` | [`rl_monitor`](tools/README.md), a live, read-only `nvtop`-style TUI to watch the memory work (tiers, cycle activity, conflicts, health): `python tools/rl_monitor.py --demo`. Plus `build_configurator.py`. |
| `resonant_lattice/DEPLOY_HERMES.md` | The exact, field-tested hermes install procedure. |
| `resonant_lattice/MODULE_MAP.md` and `MEMORY_ROADMAP.md` | Architecture and design. |

## Requirements

- **Python 3.10+**
- **`sqlite-vec`** (required, the provider declines without it) and **`numpy`** (HRR).
- **Ollama** reachable (local or LAN) for an **embedding** model (e.g. `nomic-embed-text` or
  `embeddinggemma:300m`) and a **reasoning** model for the off-hot-path consolidation (e.g.
  `deepseek-v4-flash:cloud`, a `gemma`/`nemotron` tag, or any local model).
- Optional (EXPERIMENTAL encryption tier only): `argon2-cffi`, `sqlcipher3-wheels`, OpenFHE.

```bash
pip install numpy sqlite-vec
```

---

## Agent quickstart (verify, deploy, use)

### 1. Verify it works on *your* machine

**Unit suite (no LLM needed, pure SQLite/HRR substrate, ~seconds):**
```bash
python resonant_lattice/test_resonant_lattice.py     # expect: 140 passed, 0 failed
```

**Behaviour / trust / scale suite (needs Ollama for embeddings; a few also need a reasoning
model).** Each test prints `PASS`/`FAIL` per hard invariant and writes a results file under
`results/`:
```bash
python tests/run_all.py                            # robustness battery (pinning, anti-fabrication, cross-session, conflict)
python tests/test_cross_entity_contamination.py    # right entity -> right value
python tests/test_supersession_recency.py          # current value, not stale
python tests/test_memory_poisoning.py              # pinned rule beats injected poison
python tests/test_conflict_flagging.py             # contradictions get surfaced
python tests/test_durability.py                    # concurrency + crash/restart (ACID)
python tests/scale_ceiling.py                      # recall@k + latency up to 50k live rows
python tests/stress_longhorizon.py                 # 20k facts / 50 dream cycles, bounded forgetting
python tests/test_agentic_e2e.py                   # does correct behaviour EMERGE from memory, on a real model
```
> Config via env (defaults in `tests/_common.py`): `RL_OLLAMA`, `RL_EMBED_MODEL`,
> `RL_REASON_MODEL`. Tests that can't reach Ollama exit `2` (skipped), not fail.

### 2. Deploy into hermes-agent

Full, gotcha-annotated procedure: **`resonant_lattice/DEPLOY_HERMES.md`**. The short version:
```bash
VENV=~/.hermes/hermes-agent/venv/bin
$VENV/python -m pip install numpy sqlite-vec        # into the hermes venv (NOT system python)

PLUGINS=~/.hermes/plugins                           # copy RUNTIME ONLY (dev/test scripts can hang discovery)
rsync -a --exclude-from=resonant_lattice/.deployignore --exclude .git \
      resonant_lattice/ "$PLUGINS/resonant_lattice/"

hermes config set memory.provider resonant_lattice
hermes config set plugins.resonant_lattice.embed_model  <your-embed-tag>
hermes config set plugins.resonant_lattice.reason_model <your-reason-tag>
hermes memory status     # -> Provider: resonant_lattice ... available
```
For a multi-profile install, repeat into `~/.hermes/profiles/<name>/plugins/`.

### 3. Or give a grok CLI agent this memory

`integrations/grok/` wires RLM into the xAI `grok` coding agent **out-of-loop**, through grok's
own seams: a `PreCompact` hook ingests each session into the lattice, and a `SessionStart` hook
projects the lattice into grok's native memory, so the agent wakes up already knowing it
(first-turn injection) and can search it on demand (`memory_search`), with RLM as the sole writer.
No fork of grok, no plugin API. See [`integrations/grok/README.md`](integrations/grok/README.md).

### 4. Configure for your use case

- **Tune it visually:** the [**RLM Configurator**](https://freemark.github.io/resonant-lattice-memory/)
  renders every setting with its default and ships preset profiles; it is generated from
  `config_schema.py`, so it never drifts from the code.
- Defaults are a solid, lighter core. For the full experience (gist, relations, self-model,
  narrative, importance-weighted retention) copy **`resonant_lattice/recommended_config.yaml`**.
- **Every tunable lives in one place:** `resonant_lattice/config_schema.py` (the `DEFAULTS` dict),
  which is also the `hermes memory setup` field list.
- **Time-coherent recall** (`inject_current_datetime`, on by default): a live `<current_datetime>`
  stamp is prepended to every recall injection at consumption time, so a cycle-driven agent always
  knows the real "now" without spending a tool call. Set `datetime_timezone` (an IANA zone) when
  the host runs UTC but the user lives elsewhere.
- **Pin standing rules and critical facts:** they hold as authoritative `[PRIORITY RULE]`s
  (surfaced in the `<authority_rules>` block) and are never forgotten. Turn on
  `importance_decay_discount` so high-stakes facts resist fading even when rarely recalled.

---

## What's proven (and where to see it)

Every claim below is backed by a test in this repo (`tests/`), validated on real models. Results
live in `results/` (start with `CONSOLIDATED_RESULTS.md`).

| Property | Evidence |
|---|---|
| **Recall holds at scale**, recall@1 = recall@10 = **1.0 up to ~48k live rows**, sub-linear latency | `scale_ceiling_results.md` |
| **Bounded forgetting**, 20k facts to a bounded live set; salient kept, noise pruned | `stress_report.md`, `forgetting_report.md` |
| **No cross-entity contamination**, right entity to right amount, even under load | `contamination_results.md` |
| **Current-not-stale**, value updates retained and surfaced for resolution | `recency_results.md` |
| **Poison-resistant**, a pinned rule beats a query-optimized poison; contradictions flagged | `poisoning_results.md`, `conflict_flagging_results.md` |
| **No fabrication**, source-quote attestation drops invented specifics; gist keeps exact $/IDs | `anti_fabrication_results.md`, `abstraction_fidelity_results.md` |
| **Durable**, concurrency-safe and crash/restart (SQLite ACID) | `durability_results.md` |
| **Agentic, end-to-end**, grounded recall + rule-following + poison-resistance *from memory*, on real models | `agentic_e2e_results.md` |
| **Marker A/B**, the authority tag the agent reads measurably changes obedience (two model families) | `marker_ab_*.md` |

---

## How it works (one paragraph)

A three-tier resonance store (short, mid, long) over SQLite + `sqlite-vec`, with HRR (holographic)
compositional encoding and an entity graph. Cycle-driven "dream cycles" decay, promote, abstract,
and resolve conflicts, with no wall-clock. Recall is hybrid (vector + keyword) with a precision
gate and an authority preference for pinned facts. Anti-fabrication is enforced by source-quote
attestation; the agent *influences* memory (reinforce / pin / feedback) but cannot silently
destroy it (no agent delete by default). See `resonant_lattice/MODULE_MAP.md` and
`MEMORY_ROADMAP.md` for the full design.

## Encryption tier: EXPERIMENTAL

`blind_*.py`, `crypto_keys.py`, `he_crypto.py`, `store_blind.py`, and `ENCRYPTION_ROADMAP.md`
implement an optional two-tier private store (at-rest SQLCipher + a homomorphic "blind" recall
tier, with an optional off-by-default GPU accelerator seam). It needs extra dependencies
(`sqlcipher3`, `argon2-cffi`, a real OpenFHE build) and is **not required** for the core memory
system. Treat it as a preview.

**Readiness matrix**, explicit about what each mode does and does not guarantee:

| Mode | At-rest on disk | Recall | Extra deps | Status |
|---|---|---|---|---|
| **plaintext** (default) | plaintext SQLite | full hybrid | none beyond core | stable, fully tested |
| **at-rest** | SQLCipher-encrypted (opaque without the key) | full (decrypted in memory) | `sqlcipher3`, `argon2-cffi` | experimental; at-rest byte-opacity is unit-checked |
| **blind (HE)** | **plaintext-at-rest in this build** unless composed with at-rest | homomorphic blind recall (bounded-RAM streaming scan; optional GPU seam) | a real OpenFHE build | preview; node-validated only, not a turnkey install |
| **blind + at-rest** | encrypted | blind | all of the above | **not a turnkey composition yet** |

Failure mode is **fail-closed**: if a requested encrypted binding is unavailable the provider
declines rather than silently falling back to plaintext (no false sense of encryption). Recovery
is your own key custody (`*.db.keys`): lose the key, lose the DB. Do **not** rely on the blind
tier alone for at-rest confidentiality in this build.

## License

MIT, see [LICENSE](LICENSE).
