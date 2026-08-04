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

- **v1.7.4 (2026-08-03): relation extraction and relational query, both unblocked.** Three
  silent failures, each found by measuring a live corpus rather than by an error.
  - **A model that thinks out loud no longer loses every triple.** `_llm_extract_triples`
    spliced `text[find("[") : rfind("]")+1]` into `json.loads`. A reasoning model emits a
    *complete* valid array and then keeps talking — trailing prose, or a whole second array
    after it reconsiders — so the span covered array1 + prose + array2, `json.loads` raised
    `Extra data`, and the blanket `except` returned `[]` **indistinguishably from "this fact
    states no relations"**. 8 of 23 measured calls lost everything that way.
    `first_json_array()` takes the first well-formed array via `raw_decode`, scanning
    successive `[` candidates because prose can contain one, and keeps `None` (parse
    failure) distinct from `[]` (nothing here). Re-extracting one corpus: **2,866 → 5,723
    triples, zero-relation facts 71.8% → 42.1%**, and the graph *densified* rather than
    dissolving — node recurrence 36.1% → 39.6%, largest component 82.5% → 88.0%.
  - **A question verb no longer blanks the graph.** `_QUERY_REL_KEYWORDS` is a generic
    personal-assistant set (`has`, `uses`, `lives_in`, `works_at`); a corpus with a closed
    `relation_vocabulary` shares none of it, yet a detected match still filtered the walk.
    "What **has** changed in bootstrapping research?" returned nothing while `bootstrapping`
    was a hub with 473 edges. An out-of-vocabulary detection is ignored (anchor-only
    traversal); an in-vocabulary one still filters. The store also never *received*
    `relation_vocabulary` — the query path could not know its own predicates.
  - **A hub anchor is ranked by the question, not just by confidence.** Two different
    questions on a 141-edge anchor returned the *same* triples, while the ones that actually
    answered them sat below `max_results`. Candidates are now ranked by overlap with the
    query text before truncation. Structured calls, and queries whose terms match nothing,
    are unaffected. Node suite **198 → 204**.
  - **Give a deep lattice a real recall budget.** Measured across 26 questions on the FHE
    corpus, the graph is nowhere near exhausted at the small limits a harness tends to
    inherit — and it has not saturated even at 60:

    | `limit` | triples returned | mean per question |
    |---|---|---|
    | 8 | 113 | 4.3 |
    | 25 | 317 | **12.2** |
    | 40 | 452 | 17.4 |
    | 60 | 632 | 24.3 |

    The difference is an answer, not just volume: *"Which FHE library should I use?"*
    returns **three** `implemented_in` edges (concrete, concrete-ml, tfhe) at 25 slots and
    **one** at 8. `tool_handler` already takes `limit` from the caller (default 10), so an
    agent with context to spare can ask for more today; a per-profile default is the
    follow-up. **What a bigger budget does not fix:** the count of questions returning *no*
    relations is **identical (18 of 26 answered) at every limit from 8 to 60** — the
    remaining empties are an anchoring limit, not a truncation one.
- **v1.7.3 (2026-08-02): portable relations + recall quality.**
  - **The relation layer ports across domains.** Five silent config gaps made the closed-vocabulary
    mechanism unusable outside the operational domain it was designed in (a clinical corpus enforced
    the vocabulary perfectly and still returned `[]`): `relation_subject_kinds` (the built prompt
    hardcoded "components, machines, services…"), `relation_attribute_predicates` (attribute objects
    for predicates like `reference_interval` survive strict entity binding), value-preserving
    attribute objects (no longer collapsed to the matched entity, so a `stage_defined_by` object
    keeps its numbers), alias substitution scoped to the matched span, and `relation_num_predict`
    (an uncapped relation call ran to the endpoint default and timed out into a silent empty
    result). Positional `[[s,r,o]]` triples now also parse. Defaults unchanged — existing profiles
    build a byte-identical prompt.
  - **Abbreviation-safe relational queries.** A corpus storing `HE` (hepatic encephalopathy) took
    "…He never whines." and injected liver-failure triples into a question about pain.
    `_QUERY_NON_ANCHORS` now drops pronouns/demonstratives/clause-openers as a *final* pass over
    anchors from every path, with an all-caps escape so the abbreviation itself stays queryable
    (and common proper nouns — May, Will, Mark — deliberately still anchor).
  - **A fact can no longer contradict its own quote silently (opt-in).** Attestation verified
    quote↔transcript, never content↔quote, so `99.5–102.5 °F` sat `attested` on a quote saying
    `99.8–102.8`. The new check requires a *near miss* and exempts rounding — measured on 6,640
    quoted facts: the naive rule fires 84 times (mostly legitimate unit derivations), near-miss 14,
    rounding-exempt 10, with the real contradiction caught at every tolerance. It **marks**
    `numeric_conflict`, never drops. Default off.
  - **Read-time redundancy gate (opt-in).** `recall_redundancy_ceiling`: a top-k candidate too
    cosine-close to an already-chosen result is held back and **appended**, never dropped — the
    caller still gets k, and the rank-1 answer survived every measured ceiling. Why not dedupe the
    store: a census found 16,432 pairs at or above the store's own 0.78 `similarity_threshold`
    with **65% carrying different numbers** — 0.78 is a *topic* threshold on a reference corpus,
    and merging on it would destroy multi-source variation. Node suite **186 → 198**.
- **v1.7.2 (2026-07-27): extraction integrity + observability.** A new `extraction_audit` table
  writes one row per consolidation epoch (attempts, quoteless retries, fact/quote tallies, outcome)
  outside the write lock — hermes swallows plugin stderr, so this is what makes a deployed retry
  guard evaluable at all. The quoteless retry now actually *differs* per attempt (names the
  omission, escalates temperature 0.1/0.45/0.7, capped by `quoteless_max_retries`); measured over a
  131-block campaign, `no_quote` facts fell **11.6% → 1.4%** of corpus. Abstraction unblocked:
  `_find_semantic_match(exclude_ids=)` stops a synthesis being deduped against its own inputs, and
  the abstraction passes take an explicit `embed_endpoint` so an OpenAI-shim reasoning endpoint
  can't silently insert vector-less abstractions. Also: mechanical `source_ref` recovery from tool
  results (default off), `think: false` on every reason call, and the FinalizeLock held for the
  write phase only. Node suite **164 → 186**.
- **v1.7.1 (2026-07-17): grok aperture polish.** Small surfacing + parity fixes on top of v1.7.0
  (the core changes are documentation-only; hermes runtime behavior is unchanged).
  - **Prefetch clock parity (grok).** The reference hermes provider stamps a `<current_datetime>` on
    every recall injection at consumption time (`inject_current_datetime`); grok's per-turn prefetch
    is a side-channel that carried only a machine unix timestamp. The `rlm_prefetch` MCP handler now
    stamps a model-friendly local wall clock at *serve* time, so even a cached block hands the agent a
    fresh "now" to judge open-loop staleness instead of guessing.
  - **Narrative projection surfacing.** The `## narrative` header now shows the live `memory_cycle`
    (so a per-row `now (cycle N)` reads as as-of, not the current clock), and the newest arc surfaces
    up to two `closed:` loops so a resumed-then-finished item stops reading as still open.
  - **Docs.** The dual narrative-prompt path is now explicit everywhere (freeform `narrative_prompt`
    vs opt-in `narrative_structured` JSON with typed fields) - `prompts.py`, `config_schema.py`, and
    both READMEs; the grok README prose and `entity_vocabulary` are brought current with the 18-tool /
    structured-narrative surface.
- **v1.7.0 (2026-07-17): dream / narrative enhancement.** A major upgrade to grok's autobiographical
  narrative, plus new consolidation observability.
  - **Hybrid narrative.** The session narrative is rebuilt from a hierarchical, born-fact-grounded
    HYBRID digest of the whole session - per-window born facts (the belief store's own distillation) +
    an open/decision roll-up + a head/mid/tail spine of the *actual dialogue* (both roles) - replacing
    the tail-40-episodes input that clipped long multi-window ingests. A blind cloud-judged A/B/C test
    showed the hybrid beats facts-only on coverage with no added fabrication (and that transcript-only
    is worse - it reintroduces the tail bias), so facts give the grounded skeleton and the dialogue
    spine restores the arc and secondary decisions that atomic extraction drops.
  - **Structured + temporally framed.** Narratives now carry typed fields (throughline / decisions /
    open_loops / closed / topics via an idempotent migration; `summarize_session(structured=True)` asks
    for JSON and falls back to freeform prose on a parse miss) and a `historical` flag
    (`mark_prior_narratives_historical`, a full two-direction recompute) so the newest reads as current
    status. Every stored narrative is ASCII-guaranteed. The projection renders the newest arc in full +
    older as one-liners, ordered by cycle; a new read-only `rlm_narrative` MCP tool reads past the top-N.
  - **Dream observability.** A new read-only `rlm_dream` MCP tool + node script surfaces consolidation
    health - tier flow (short/mid/long) with promotion-ready counts, dwell maturity, decay/fading,
    contested facts, abstraction/gist output, and the dials in effect (each count beside the threshold
    that governs it).
  - **Anti-fossil hygiene.** The extraction prompt now rejects the lattice's own transient state
    readings (cycle counters, fact/tier counts, before/after tallies, health snapshots) as non-durable.
  - MCP surface **16 -> 18 tools** (`rlm_narrative`, `rlm_dream`). Backward-compatible throughout: the
    structured/digest paths are opt-in, so the reference hermes narrative path is unchanged. Node test
    suite **140 -> 146**.
- **Earlier (one-liners; full detail in the release notes):**
  - **v1.6.9** grok narrative spans the whole session: hierarchical born-fact digest replaces the tail-40-episode cap.
  - **v1.6.8** grok re-compact dedup: per-session ingest high-water mark, only the new tail is re-mined.
  - **v1.6.7** grok compact/ingest observability: `rlm_watch_ingest.py`, unbuffered ingest logging.
  - **v1.6.6** grok transport stdin fix: input-less node calls no longer swallow the MCP JSON-RPC pipe.
  - **v1.6.5** grok per-turn prefetch: staged `<resonant_memory>` per user turn, `rlm_prefetch`, reinforce-on-use.
  - **v1.6.4** concurrent MCP dispatch: per-call worker threads, no head-of-line blocking.
  - **v1.6.3** local transport by default for the grok hooks; ssh only when `SSH_HOST` is set.
  - **v1.6.2** `entity_vocabulary`: domain terms become graph entities; snake_case patterns widened.
  - **v1.6.1** `relation_extract_from_transcript` (default off): mine relations from the raw ingest window.
  - **v1.6.0** domain-configurable relation graph: closed `relation_vocabulary`, `entity_aliases`, `rlm_relational` + `rlm_infer`.
  - **v1.5.x** the grok memory lifecycle completes: hybrid search over its own lattice + read-only external domain lattices + `transfer_knowledge`; write/curation verbs (`rlm_pin`/`rlm_unpin`/`rlm_remember`/`rlm_forget`); dreaming + rolling narrative per ingest; `rlm_stats`/`rlm_conflict`/`rlm_feedback`/`rlm_inspect`/`rlm_entity`/`rlm_self_model`; ACP transcript parsing.
  - **v1.4.x** the grok integration is born (PreCompact ingest + SessionStart projection, out-of-loop, RLM sole writer); operator-hardening (`<authority_rules>`, prompt dials, conflict-detection fixes); GPU blind-recall seam (off by default); bounded-RAM blind recall (5.0 GB -> 0.65 GB on a 4,952-fact corpus); train-in-the-open, seal-after-the-fact.

Full history in the [Releases](https://github.com/FreeMark/resonant-lattice-memory/releases).

---

## What's in this repo

| Path | What it is |
|---|---|
| `resonant_lattice/` | The plugin (runtime code, `plugin.yaml`, `recommended_config.yaml`, architecture docs, the **207-test** unit suite, the eval harness). |
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
python resonant_lattice/test_resonant_lattice.py     # expect: 207 passed, 0 failed
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
- **Narrative has two prompt paths:** `narrative_prompt` writes the freeform one-paragraph gist (the
  default). Opt in to `narrative_structured: true` (+ optional `narrative_structured_prompt`) for the
  structured path, which asks the model for JSON and fills typed fields
  (throughline / decisions / open_loops / closed / topics), falling back to freeform on a parse miss.
  In structured mode the freeform `narrative_prompt` is not used.
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
