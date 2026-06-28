# Resonant Lattice Memory

A **neuroplastic, local-first long-term memory** plugin for AI agents (built for
[hermes-agent](https://hermes-agent.nousresearch.com), usable standalone). Facts
behave like resonant circuits: a memory that keeps getting struck stays in tune
and rings loud (reinforced → promoted to a durable tier); the ones nothing
resonates with fade to silence (decay → prune). The result is a memory that
**keeps what matters and forgets the noise — on its own, driven by usage cycles,
not wall-clock timers.**

It is designed so **an agent can read this file, verify the system on its own
machine, deploy it, and configure it** — the same loop a human + agent used to
build and harden it.

> **Thesis:** for an agent that earns, spends, or runs real operations, *memory is
> the load-bearing wall.* It must (1) recall the right thing across sessions,
> (2) never fabricate a fact, (3) obey standing rules, and (4) resist being
> poisoned or going stale. This system is built and **tested** around those four.

---

## What's in this repo

| Path | What it is |
|---|---|
| `resonant_lattice/` | The plugin (runtime code, `plugin.yaml`, `recommended_config.yaml`, architecture docs, the 97-test unit suite, the eval harness). |
| `tests/` | The test suite (substrate → behaviour → scale → durability), plus the live end-to-end exercise `live_e2e.py`. |
| `results/` | All test evidence in one place: per-test outputs, metrics (`.jsonl`), model-comparison summaries, and the single-file [`CONSOLIDATED_RESULTS.md`](results/CONSOLIDATED_RESULTS.md). |
| `resonant_lattice/DEPLOY_HERMES.md` | The exact, field-tested hermes install procedure. |
| `resonant_lattice/MODULE_MAP.md` · `MEMORY_ROADMAP.md` | Architecture + design. |

## Requirements

- **Python 3.10+**
- **`sqlite-vec`** (required — the provider declines without it) and **`numpy`** (HRR).
- **Ollama** reachable (local or LAN) for an **embedding** model (e.g. `nomic-embed-text`
  or `embeddinggemma:300m`) and a **reasoning** model for the off-hot-path consolidation
  (e.g. `deepseek-v4-flash:cloud`, `gemma`/`nemotron`, or any local model).
- Optional (EXPERIMENTAL encryption tier only): `argon2-cffi`, `sqlcipher3-wheels`, OpenFHE.

```bash
pip install numpy sqlite-vec
```

---

## Agent quickstart (verify → deploy → use)

### 1. Verify it works on *your* machine

**Unit suite (no LLM needed — pure SQLite/HRR substrate, ~seconds):**
```bash
python resonant_lattice/test_resonant_lattice.py     # expect: 97 passed, 0 failed
```

**Behaviour / trust / scale suite (needs Ollama for embeddings; a few also need a
reasoning model).** Each test prints `PASS`/`FAIL` per hard invariant and writes a
results file under `results/`:
```bash
# the corrected business-robustness battery (pinning, anti-fabrication, cross-session, …)
python tests/run_all.py

# the trust axis — does the agent recall the right thing and refuse the wrong thing?
python tests/test_cross_entity_contamination.py   # right entity → right value
python tests/test_supersession_recency.py         # current value, not stale
python tests/test_memory_poisoning.py             # pinned rule beats injected poison
python tests/test_conflict_flagging.py            # contradictions get surfaced
python tests/test_durability.py                   # concurrency + crash/restart (ACID)

# scale + retention (longer; configurable via env, see each file's header)
python tests/scale_ceiling.py        # recall@k + latency at up to 50k live rows
python tests/stress_longhorizon.py   # 20k facts / 50 dream cycles, bounded forgetting
python tests/forgetting_probe.py     # the fade curve (use-it-or-lose-it)

# end-to-end: does correct behaviour EMERGE from memory, on a real model?
python tests/test_agentic_e2e.py     # set RL_AGENT_* envs to your agent model/endpoint
```
> Config via env (defaults in `tests/_common.py`): `RL_OLLAMA`,
> `RL_EMBED_MODEL`, `RL_REASON_MODEL`. The agentic/marker tests take an agent
> model+endpoint too. Tests that can't reach Ollama exit `2` (skipped), not fail.

### 2. Deploy into hermes-agent

Full, gotcha-annotated procedure: **`resonant_lattice/DEPLOY_HERMES.md`**. The short version:
```bash
VENV=~/.hermes/hermes-agent/venv/bin
$VENV/python -m pip install numpy sqlite-vec        # into the hermes venv (NOT system python)

# copy RUNTIME ONLY into the plugins dir (dev/test scripts can hang hermes discovery)
PLUGINS=~/.hermes/plugins
rsync -a --exclude-from=resonant_lattice/.deployignore --exclude .git \
      resonant_lattice/ "$PLUGINS/resonant_lattice/"   # (or cp -r then delete test_*/eval_*)

hermes config set memory.provider resonant_lattice
hermes config set plugins.resonant_lattice.embed_model  <your-embed-tag>
hermes config set plugins.resonant_lattice.reason_model <your-reason-tag>
hermes config set plugins.resonant_lattice.ollama_endpoint_embed  http://localhost:11434
hermes config set plugins.resonant_lattice.ollama_endpoint_reason http://localhost:11434
hermes memory status     # → Provider: resonant_lattice … available ✓
```
For a multi-profile install, repeat into `~/.hermes/profiles/<name>/plugins/`.

### 3. Configure for your use case

- Defaults are a solid, lighter core. For the full experience (gist, relations,
  self-model, narrative, importance-weighted retention) copy **`resonant_lattice/recommended_config.yaml`**.
- **Every tunable lives in one place:** `resonant_lattice/config_schema.py` (the
  `DEFAULTS` dict). It's also the `hermes memory setup` field list.
- For a **money/compliance agent**: pin policies (they hold as authoritative
  `[PRIORITY RULE]`s), and turn on `importance_decay_discount` so high-stakes facts
  resist fading even when rarely recalled.

---

## What's proven (and where to see it)

Every claim below is backed by a test in this repo (`tests/`), validated
on real models. Results live in `results/` (start with `CONSOLIDATED_RESULTS.md`).

| Property | Evidence |
|---|---|
| **Recall holds at scale** — recall@1 = recall@10 = **1.0 up to ~48k live rows**, sub-linear latency | `scale_ceiling_results.md` |
| **Bounded forgetting** — 20k facts → bounded live set, real plateau; salient kept, noise pruned | `stress_report.md`, `forgetting_report.md` |
| **No cross-entity contamination** — right entity → right amount, even under load | `contamination_results.md` |
| **Current-not-stale** — value updates retained + surfaced for resolution | `recency_results.md` |
| **Poison-resistant** — a pinned rule beats a query-optimized poison; contradictions flagged | `poisoning_results.md`, `conflict_flagging_results.md` |
| **No fabrication** — source-quote attestation drops invented specifics; gist keeps exact $/IDs | `anti_fabrication_results.md`, `abstraction_fidelity_results.md` |
| **Durable** — concurrency-safe + crash/restart (SQLite ACID) | `durability_results.md` |
| **Agentic, end-to-end** — grounded recall + rule-following + poison-resistance *from memory*, on real models | `agentic_e2e_results.md` |
| **Marker A/B** — the authority tag the agent reads measurably changes obedience (validated on two model families) | `marker_ab_*.md` |

---

## How it works (one paragraph)

A three-tier resonance store (short → mid → long) over SQLite + `sqlite-vec`, with
HRR (holographic) compositional encoding and an entity graph. Cycle-driven "dream
cycles" decay, promote, abstract, and resolve conflicts — no wall-clock. Recall is
hybrid (vector + keyword) with a precision gate and an authority preference for
pinned facts. Anti-fabrication is enforced by source-quote attestation; the agent
*influences* memory (reinforce / pin / feedback) but cannot silently destroy it
(no agent delete by default). See `resonant_lattice/MODULE_MAP.md` and
`MEMORY_ROADMAP.md` for the full design.

## Encryption tier — EXPERIMENTAL

`blind_*.py`, `crypto_keys.py`, `he_crypto.py`, `store_blind.py` and
`ENCRYPTION_ROADMAP.md` implement an optional two-tier private store (at-rest
SQLCipher + a homomorphic "blind" recall tier). It needs extra dependencies
(`sqlcipher3`, `argon2-cffi`, a real OpenFHE build) and is **not required** for the
core memory system. Treat it as a preview.

## License

MIT — see [LICENSE](LICENSE).
