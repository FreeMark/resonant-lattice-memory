# Resonant Lattice — Module Map

Map of the current flat-module structure. It began as a behaviour-preserving split of two
former "god files" (`store.py`, `__init__.py`) into flat sibling mixins (see `REFACTOR_NOTES.md`),
and has since grown the 8-phase memory roadmap (P1–P8) and the encryption north star (E0–E7 +
provider wiring; see `ENCRYPTION_ROADMAP.md`). A full remediation (see `REMEDIATION_PLAN.md`) was performed
for consistency, hygiene, resilience, and documentation. No subpackages; all sibling imports are flat.
Line counts below are approximate and drift — they orient, they aren't a contract.

## Load-path contract (unchanged)

- **Plugin loader:** imports the package dir and calls `register(ctx)` (or finds
  a top-level `MemoryProvider` subclass). `__init__.py` still exposes a working
  `register(ctx)` and the `LatticeMemoryProvider` class. The `name` property is
  still `"resonant_lattice"`.
- **Test harness:** loads modules by **bare filename** via
  `spec_from_file_location` after putting the plugin dir on `sys.path`. All
  sibling imports are **flat** (`from store_facts import FactsMixin`), never
  package-relative. `store.py`, `retrieval.py`, `holographic.py`,
  `entity_extractor.py` remain importable by those exact names.

## Store side — `LatticeStore` (composite of six mixins)

`store.py` keeps `__init__`, the cycle counters, `get_stats`, `close`, and
composes:

```
class LatticeStore(SchemaMixin, FactsMixin, DreamCycleMixin,
                   AbstractionMixin, EpisodesMixin, EntitiesMixin,
                   RelationsMixin, IdentityMixin, NarrativeMixin)
```

| File | Class | Responsibility |
|------|-------|----------------|
| `store.py` (356) | `LatticeStore` | construction, `serialize_vector`/`_ENTITY_EXTRACTOR_AVAILABLE` re-exports, cycle counters, `get_stats`, `close` |
| `store_common.py` (120) | — | shared leaf: `serialize_vector`, the `pysqlite3`-fallback `sqlite3`, and the `holographic`/`entity_extractor` optional-import blocks (`hrr`, `_HRR_AVAILABLE`, `_extract_entities_fn`, `_ENTITY_EXTRACTOR_AVAILABLE`). **Dependency-free of `LatticeStore`** so mixins never circular-import the composite. |
| `store_schema.py` (865) | `SchemaMixin` | `_init_db`, all `_migrate_*` (incl. the `semantic_he*` + `reencrypt_audit` blind tables), `_stamp_meta`, `_validate_vector_dim` |
| `store_facts.py` (429) | `FactsMixin` | `add_or_reinforce_fact`, `_find_semantic_match`, `_reinforce_fact`, `_link_entities`, `get_fact`, `get_facts_for_entity`, `adjust_resonance`, `reinforce_on_recall`, `set_pinned` (P4a A5 never-forget), `remove_fact`, and the blind read-back helpers `get_fact_embedding`/`get_fact_hrr_phases` |
| `store_dream.py` (701) | `DreamCycleMixin` | decay, dwell, promotion, conflict bleed/resolution, pruning (all pinned-exempt, P4a), long-tier cap, HRR re-encode, `reembed_if_needed` (P4d embed_model-switch migration), `_phases_from_blob` |
| `store_abstraction.py` (793) | `AbstractionMixin` | `perform_abstraction_pass`, `_get_embedding_for_abstraction`, `get_abstraction_sources`, `get_abstraction_explanation`, `distill_procedural_facts`, `consolidate_before_prune`, `_clean_llm_json` |
| `store_episodes.py` (186) | `EpisodesMixin` | conversational + tool episode logs and their pruning |
| `store_entities.py` (101) | `EntitiesMixin` | `_extract_entities`, `gc_orphan_entities`, `get_entities_for_fact`, `get_related_entities` |
| `store_relations.py` (705) | `RelationsMixin` | Phase 5a: `extract_triples` (deterministic, entity-grounded), `_llm_extract_triples` (optional, default-off), `_encode_triple_blob`, `store_fact_relations`, `extract_and_store_relations`, `get_fact_relations`, `get_relations`. Phase 5b: `relational_recall` (graph SQL + HRR partial-probe fuzzy), `_parse_relational_query`. Phase 5c: `infer_relations` (bounded transitive chaining; never writes), `_compose_inference` |
| `store_identity.py` (123) | `IdentityMixin` | Phase 7 deliberate self-model: `set_self_model`, `get_self_model`, `delete_self_model`, `seed_self_model` over the separate `agent_identity` table (autonomous ingest can never reach it) |
| `store_narrative.py` (150) | `NarrativeMixin` | Phase 8 autobiographical layer: `add_session_summary`, `get_recent_narrative`, `prune_session_summaries`, `summarize_session` (LLM gist of a session) over the durable `session_summaries` table (survives episode pruning) |
| `store_blind.py` (BlindMixin) | `BlindMixin` | STORE side of the blind tier — opaque-blob CRUD over `semantic_he*` + the `reencrypt_audit` log + the `facts_missing_blind` reconciliation worklist. See the Encryption section below. |

`retrieval.py` gains the blind classes (see below); `entity_extractor.py` is untouched.
`holographic.py` gains `encode_triple` / `encode_triple_query` (Phase-5b relational) and
`hrr_lift` (E4 blind HRR).

## Provider side — `LatticeMemoryProvider` (composite of four mixins)

```
class LatticeMemoryProvider(ToolHandlerMixin, ConsolidationMixin, RecallMixin,
                            LifecycleMixin, MemoryProvider)
```

`__init__.py` keeps construction/identity (`__init__`, `name`, `is_available`,
`initialize`, `_probe_vector_dim`, `_read_db_vector_dim`, `system_prompt_block`,
`sync_turn`, `_is_self_referential_infra`, `get_config_schema`, `save_config`),
plus `register(ctx)` and `_load_plugin_config`. It re-imports every moved
module-level name so the package's import surface is unchanged.

| File | Class / contents | Responsibility |
|------|------------------|----------------|
| `__init__.py` (1026) | `LatticeMemoryProvider`, `register`, `_load_plugin_config` | thin composite/entry point + core lifecycle/construction + the `_resolve_blind_*` setup methods |
| `attestation.py` (137) | `_attest_source_quote` (+ `_normalize_for_match`, `_digit_core`, `_QUOTE_NUM_TOKEN_RE`) | pure two-channel source_quote verifier (re + difflib only; the test loads this directly) |
| `self_write_gate.py` (78) | `is_self_referential_infra` + denylists | Phase-E self-write policy boundary |
| `prompts.py` (99) | `DEFAULT_EXTRACTION_PROMPT`, `DEFAULT_CONSOLIDATION_PROMPT`, `DEFAULT_PROCEDURAL_PROMPT`, `DEFAULT_GIST_PROMPT`, `DEFAULT_RELATION_PROMPT`, `DEFAULT_NARRATIVE_PROMPT` | default LLM prompt strings (text only) |
| `config_schema.py` (273) | `CONFIG_SCHEMA` | static `hermes memory setup` field list |
| `tool_handler.py` (355) | `ToolHandlerMixin` + `LATTICE_STORE_SCHEMA` | `get_tool_schemas`, `handle_tool_call` (P4a: no agent `delete` — gated behind `agent_can_delete`, steered to `feedback`; + `pin`/`unpin`/`request_abstraction`). **Named `tool_handler`, not `tools`**, to avoid shadowing Hermes' `tools` package on `sys.path[0]`. |
| `consolidation.py` (716) | `ConsolidationMixin` | waking epoch, dream cycle, abstraction/distillation kicks, tool-action ingest, `_attest_quote`, `_reembed_if_needed` (P4d embed_model-switch), `_blind_reconcile` (blind write-path completeness) |
| `recall.py` (150) | `RecallMixin` | `prefetch`, `queue_prefetch`, recall reinforcement, `<resonant_memory>` block (P4b: surfaces A22 peak / entry-cycle / `[PINNED]`) |
| `lifecycle.py` (223) | `LifecycleMixin` | session switch/end, shutdown, `on_memory_write`/`on_pre_compress`/`on_delegation` |

## Encryption / blind tier (branch `encryption-northstar` — see `ENCRYPTION_ROADMAP.md`)

Added on the `encryption-northstar` branch (not in the structural-refactor baseline above).
Two tiers under `encryption_mode` (`none` default | `at_rest` | `blind`). The **helper layer**
(below) is node-validated at the ciphertext/SQLite substrate and UNTRACKED by inventory/
verify_logic; the **provider glue** is harness-validated only (Hermes isn't installable on the
dev box/node), but every component it wires is node-proven.

| File | Class / contents | Responsibility |
|------|------------------|----------------|
| `crypto_keys.py` | KDF + keystores (no class) | Argon2id passphrase→master→HKDF subkeys; at-rest SQLCipher key (`derive_db_key`); HE secret AES-GCM wrap/unwrap. **Single-keyset** `setup_or_load_blind_client`. **Multi-keyset (Option A, 2a)** `setup_or_load_blind_contexts` + `create/load_multi_he_keystore` / `multi_he_key_blobs_from_keystore` / `multi_he_keystore_is_secret_free` (named keysets recall/hrr/maint). Entity AEAD (`derive_entity_key`, `encrypt_entities`, `decrypt_entities`). |
| `he_crypto.py` | `BlindRecallPRE`, `BlindMaintenance`, `BlindArgmaxCKKS` (+ co-located refs `BlindCrypto`/`BlindArgmax`/`BlindPRE`/`ThresholdAudit`) | OpenFHE CKKS engines. `BlindRecallPRE` = E2 cosine + E6 PRE (3 roles). `BlindMaintenance` = decay + Chebyshev threshold; `generate(depth=…)`, `_MAINT_BLIND_DEPTH=1` light decay-only. `BlindArgmaxCKKS` = pure-CKKS argmax. Eval keys serialized BY KEYPAIR TAG so keysets coexist in one process. Inert without `openfhe`. |
| `store_blind.py` | `BlindMixin` | STORE side: opaque-blob CRUD over `semantic_he` (embedding) / `semantic_he_hrr` (HRR lift) / `semantic_he_meta` (resonance) / `semantic_he_entities` (AEAD entity set) via an allowlisted `table` selector; `reencrypt_audit` log. No `openfhe` (the store never decrypts). |
| `retrieval.py` (blind) | `BlindRetriever`, `BlindWriter`, `BlindMaintainer`, `BlindEntityStore` | CLIENT side. `BlindRetriever`: blind embedding recall + blind HRR recall (separate `blind_hrr` 2·hrr_dim client). `BlindWriter(table=…)`: encrypt→store. `BlindMaintainer`: blind decay + client settle. `BlindEntityStore`: client-side entity overlap/conflicts over AEAD sets. Duck-typed on the he_crypto clients (no `openfhe` import). |
| `blind_policy.py` | `ScopeLimiter`, `BlindReEncryptGate`, `ReEncryptAuditLog` | §7.2 honest-seam policy: per-cycle scope caps + single-use re-encryption token gate + audit. Pure Python. |
| `holographic.py` (+) | `hrr_lift` | `(cos φ, sin φ)/√dim` L2-unit lift so HRR similarity = the E2 inner product (E4 4a). |

**Provider glue (`__init__.py` / `consolidation.py`).** `_resolve_encryption_db_key` (E0 at-rest);
`_resolve_blind_contexts` (Option A multi-keyset → `{recall, hrr, maint}`; supersedes the single-keyset
`_resolve_blind_client`); `_resolve_blind_entities` (AEAD entity store, openfhe-free). `initialize`
sets `self._blind`/`_blind_hrr`/`_blind_maint` + `self._blind_writer`/`_blind_hrr_writer`/`_blind_entities`
+ swaps in `BlindRetriever` under `encryption_mode=blind`. `_run_consolidation_epoch` mirrors each fact's
embedding + entity set + HRR lift into the encrypted tables. **Schema** (`store_schema.py`): table-only
migrations `_migrate_add_semantic_he`/`_hrr`/`_meta`/`_entities` + `_migrate_add_reencrypt_audit`.
**Status / next** (write-path completeness, etc.): `ENCRYPTION_ROADMAP.md` §14 Priority 6.

## Mixin discipline

Mixins never import the composite class; they rely on attributes the composite
defines (`self._conn`, `self._lock`, config attrs) and call sibling methods via
`self` (resolved through the MRO). Shared module-level helpers come from
`store_common.py` (store side) or the pure leaf modules `attestation.py` /
`self_write_gate.py` (provider side). The HRR import block is shared via
`store_common` rather than duplicated.

## Verification (every change)

Run from the repo root. Current green: **97 unit tests**. The encryption/HE tests
self-skip without `openfhe`/`cryptography`; HE crypto is node-validated separately.

- `python resonant_lattice/test_resonant_lattice.py` — the unit suite (store, provider,
  HE/crypto, evals), validated at the SQLite substrate.
- `python tests/stub_loader.py` — stub Hermes loader: the provider loads + registers
  (`available=True`, tools present, a live `memory_audit` action) without the real framework.
- `python tests/live_e2e.py --model <tag>` — real store + retriever end-to-end against
  Ollama, validated at the SQLite substrate.
- The blind tier's helper layer is node-validated over SSH (the "have the files been
  synced — yes/no" workflow); the provider glue is harness-validated only (Hermes isn't
  installable on the dev box/node). See `ENCRYPTION_ROADMAP.md` §10/§14.
