# Resonant Lattice — Encryption Roadmap (the "blind store" north star)

A phased plan to give the memory system a **master-passphrase-controlled encrypted
substrate**, with an opt-in **homomorphic "blind store"** mode for running memory on
hardware the user does not fully trust. This is a larger architectural effort than any
single memory-roadmap phase; like that roadmap, we work it one gated, test-backed,
substrate-validated phase at a time.

---

> **STATUS (2026-06-19, branch `encryption-northstar`, HEAD `c670322`).** Tier 0 (E0 encrypted-at-rest,
> SQLCipher whole-DB) shipped + substrate-validated. **Tier 1 (HE blind store) is a real switch-on-able
> tier** (`encryption_mode=blind`, default `none`). The numbered phase CORES E0–E7 are all done +
> node/substrate-validated (per-phase detail + commit refs in §14): **0a** serializable purpose-built
> engines + FHEW→pure-CKKS pivot; **0b/0c** HE keystore + `BlindWriter` + blind write/recall wiring;
> **E3 §3b** production 128-bit; **E6 6c** token-gated re-encryption + persisted audit; **E4 4a/4b** HRR
> via the (cos,sin) lift; **E5 5a/5b** decay + threshold compare (decay blind / promote-evict client-
> assisted); **E7 7a/7b** drop-FTS + AEAD entity sets.
>
> **NOW: PROVIDER WIRING (routing the blind helpers into the live `LatticeMemoryProvider`) — in progress
> (§14 "Priority 6"):** **Phase 1** blind entity-set write mirror (`0b4fecc`); **2a** multi-keyset HE
> keystore (Option A: recall@embed-dim + HRR@2·hrr-dim + light decay-only maint) + by-tag eval-key
> serialization (`e179bc1`, node-val); **2b-i** multi-context `initialize` + HRR write mirror (`c4d074a`);
> **2b-ii(a)** blind HRR recall uses the separate 2·hrr_dim client (`d378488`, node-val). So a
> consolidation-created fact now mirrors its embedding + entity set + HRR lift into the encrypted tables
> on write. **6a write-path completeness DONE** (`c670322`): one idempotent `_blind_reconcile()` pass
> (end of consolidation epoch + dream cycle) mirrors EVERY fact-creation path — abstraction/gist/
> procedural (store-side) + builtin + a first-blind-enable backfill — by reading embedding/HRR/entities
> back from the plaintext store (no Ollama); node-validated (backfill → blind recall == plaintext).
> **NEXT: 2b-ii(b)** resonance + dream-cycle blind maintenance (DEFERRED pending the §5 source-of-truth
> decision; decay-from-origin at depth 1), then **6e** PRE-runtime recall + a `memory_audit` action over
> the re-encryption log, and the full §5 blind store (AEAD content/source). **Validation seam:** the helper layer (he_crypto/crypto_keys/retrieval/
> blind_policy/holographic) is node-validated at the substrate; the provider glue is harness-validated
> only (Hermes isn't installable on the dev box/node) — every wired component is node-proven.

---

## 1. The goal, in one paragraph

The user controls a single **master secret derived from a passphrase at `hermes memory
setup`**. From that one secret the system protects the memory store. Two tiers, selected
at setup:

- **Tier 0 — Encrypted-at-rest (baseline).** The whole memory DB (and WAL, and backups)
  is opaque ciphertext on disk; only a process holding the passphrase-derived key can
  read it. The running process decrypts in RAM, so **all functionality is preserved**
  (vector search, FTS5, entity overlap, dream cycle) at near-zero cost. This delivers
  the literal goal — "an encrypted database the user controls the master key for."
- **Tier 1 — Homomorphic blind store (opt-in, "extremely secure").** The store node
  holds **only** a public/evaluation key and computes on ciphertext it can never read;
  the agent holds a query-bound decryption capability; the user re-derives the master
  secret for full audit/export. This is the tier that earns its enormous cost when the
  store runs on a **parallel inference node or external hardware** the user does not
  fully trust. Enabled with a single setup flag; off by default.

The tiers compose: a Tier-1 deployment can still be Tier-0-wrapped at rest on the
untrusted node as defense in depth.

## 2. Why two tiers (the trust-boundary analysis)

HE exists for exactly one purpose: **to let an untrusted party compute on data it cannot
read.** Its cost is only "paid for" when there is a real boundary between who holds the
keys and who runs the compute. The deployment decides which tier is appropriate:

| Adversary / scenario | Tier 0 (at-rest) | Tier 1 (HE blind store) |
|---|---|---|
| Stolen disk / DB file / backup leak, process OFF | ✅ defended | ✅ defended |
| Store runs on a node the user doesn't control (RAM, operator, disk) | ❌ plaintext in RAM | ✅ defended — store never holds the key |
| **Compromised *agent* process at runtime** | ❌ | ❌ *(see §7 tension #2 — HE does NOT fix this)* |
| Local single-trusted-workstation, store == agent process | (over-engineering) | (over-engineering) |

The user's deployment is **"local but on a parallel inference node,"** trending toward
**"runnable on external/blind hardware."** That is a genuine untrusted-compute boundary,
so Tier 1 is justified — but only for the boundary between the (untrusted) store node and
the (trusted) agent + user. Tier 1 explicitly does **not** defend a compromised agent
(it holds the eval key and can issue queries); §7 addresses how we bound that blast
radius with scope limits rather than pretending HE solves it.

## 3. Cross-cutting principles (inherited + encryption-specific)

Inherit all six MEMORY_ROADMAP principles (cycle-driven not wall-clock; idempotent
self-detecting migrations; anti-fabrication is sacred; land on the mixin structure;
behaviour behind a flag; substrate-validated acceptance). Add:

1. **The master secret never touches disk.** Passphrase → Argon2id KDF → master secret
   in an `mlock`'d buffer (no swap; `MADV_DONTDUMP` where available), zeroized
   immediately after key derivation/destructuring at setup. Only derived sub-keys and
   public/eval material persist, and only as the tier requires.
2. **The store decrypts nothing in Tier 1.** Under HE the store process must be
   structurally incapable of plaintext: it is given the public + eval key only. Any code
   path that would require the secret key on the store side is a design bug.
3. **Plaintext-only operations move to the trusted side.** Embedding (Ollama
   `nomic-embed-text`), entity extraction, HRR encoding, and source-quote attestation
   are inherently plaintext. In Tier 1 they run **client/agent-side at write time**; the
   store only ever receives ciphertext (encrypted vectors, encrypted content, encrypted
   entity sets). This is the resolution of FTS/entity tension #4 (§7).
4. **Cycles, not seconds, absorbs the slowdown.** The system is not latency-bound;
   dream-cycle maintenance and recall run on the logical clock, so we can afford
   homomorphic ops where a realtime system could not. This is the philosophical
   synergy that makes Tier 1 feasible at all.
5. **Reversible & opt-in.** Tier 0 is wrappable/unwrappable via export→re-key→import.
   Tier 1 is a setup choice, never forced; a fresh store defaults to plaintext (today's
   behaviour) unless the user opts into Tier 0 or Tier 1.
6. **Heavy crypto deps are isolated to the tier that needs them.** The default + Tier-0
   path use only light, portable deps (`cryptography` / `argon2-cffi`, or SQLCipher).
   The HE native dependency (§8) is imported **only** when Tier 1 is enabled, so the
   "no new heavy deps" guardrail holds for every non-HE deployment.

## 4. Key hierarchy & setup flow

```
passphrase  ──Argon2id(salt, mem/time params)──►  MASTER SECRET  (mlock'd, ephemeral)
                                                        │
            ┌───────────────────────────────────────────┼───────────────────────────────┐
            ▼ (Tier 0)                                    ▼ (Tier 1, if enabled)
   AEAD data key (HKDF ctx="rest")            HE keypair (CKKS, generated from master-seeded CSPRNG)
   → encrypts DB at rest                        ├─ public key      → store + agent (encrypt)
                                                 ├─ eval/relin/rot keys → store + agent (homomorphic ops)
                                                 ├─ runtime decrypt capability → agent (query-bound; §7.1)
                                                 └─ secret key      → reconstructable ONLY from passphrase
                                                                       (user god-mode audit/export)
```

- **Salt + KDF params + public/eval material** persist in a small sidecar keystore
  (e.g. `lattice.keys` next to the DB) — none of it secret. The secret key and master
  secret are **never** stored; they are re-derived from the passphrase on demand.
- **Setup (`hermes memory setup`)** gains: prompt for passphrase (confirm), choose tier,
  derive master, generate the tier material, **destroy** the master + passphrase buffers,
  persist only the keystore. One-time; re-runnable to rotate (export→re-key→import).
- **Open keys at process start:** Tier 0 re-derives the AEAD key from the passphrase
  (prompted, or supplied via an agreed secure channel — see §11). Tier 1 store loads only
  public/eval keys (no prompt needed on the store node — that's the point).
- **Recovery / loss:** lose the passphrase → lose the data (sovereign by design). Setup
  must say this in the clearest possible terms and offer an optional user-held recovery
  export.

### 4.1 The three keys in plain terms (the user's mental model, validated)

The desired property — *"the agent can use memory at runtime without the passphrase, but
its key does not grant full decryption of the database"* — is a genuine, achievable
**Tier-1 / PRE** property (it is NOT achievable in Tier 0, whose runtime key, if it can
decrypt to use the data, can decrypt all of it). The key that makes it work: **the agent's
use-key operates on query *results* the store re-encrypts for it — never on the stored
rows.**

1. **Master secret** (from passphrase, setup-only) → full decrypt / god-mode. Never on disk.
2. **Store public + eval key** (on the inference node) → encrypts new facts, runs blind
   homomorphic search. Structurally cannot decrypt anything.
3. **Agent use-key**, via a one-time **re-encryption key `rk_master→agent`** generated at
   setup while the passphrase is present → decrypts ONLY results the store re-encrypts for a
   query the agent ran. Applied to the raw DB it decrypts NOTHING.

Runtime flow, no passphrase present: agent encrypts query (public key) → store
homomorphically searches the encrypted DB (cannot read it) → store applies `rk_master→agent`
to flip the result from the master key to the agent key (still cannot read it) → agent
decrypts that one result with its use-key → LLM uses plaintext. Honest seam (= tension #2):
a *hijacked* agent can still ask the store to re-encrypt results query-by-query and
reassemble memory; HE forbids decrypting the DB at rest, but the "no exfiltration via
queries" bound is **policy (scope/rate caps + re-encryption audit log), not math.**

### 4.2 Key-at-boot for the headless Linux server (resolves §11 #1)

Setup is the only time the passphrase exists: prompt → derive master → generate {store
public/eval key, `rk_master→agent`, agent use-key} → **destroy passphrase + master**. The
unattended server then needs the agent use-key + store keys at boot without a human.
Chosen model, strongest→simplest: **(a) TPM2-sealed** (chip releases the use-key only to the
expected boot state; default if the server has a TPM); **(b) unlock-once-per-boot over SSH**
(server boots locked; user SSHes in once to unseal the use-key into RAM/tmpfs for the run —
portable fallback, matches the sovereignty instinct); **(c) permission-locked sealed file**
(simplest, weakest). The passphrase is needed again only for god-mode / recovery / rotation.

## 5. Encryption surface — what actually gets protected

From the live schema (`store_schema.py`) and write path (`store_facts.add_or_reinforce_fact`):

| Data | Where | Tier 0 (at-rest) | Tier 1 (blind store) |
|---|---|---|---|
| `semantic_facts.content` (plaintext, UNIQUE) | row | transparent (whole-DB) | AEAD ciphertext; UNIQUE→deterministic content hash; dedup reworked |
| `semantic_facts.hrr_vector` (float64 phases) | blob | transparent | CKKS ciphertext (phase-add/similarity are HE-friendly) |
| `semantic_vec.embedding` (768-d nomic) | vec0 | transparent | **cannot use sqlite-vec** — homomorphic dot-product instead (§7.3, perf crux) |
| `semantic_fts` (FTS5 over content/category) | fts5 | transparent | **dies** — no full-text over ciphertext; client-side or SSE index (§7.4) |
| `entities.name` + `fact_entities` | rows | transparent | encrypted entity sets; overlap via PSI / encrypted-set ops, client-side at write |
| `source_quote`, `source_ref`, `category`, triples, `session_summaries`, `agent_identity` | rows | transparent | AEAD ciphertext |
| `meta` cycle clock, resonance_count, tiers, counts | rows | transparent | scalars stay BFV/CKKS-encryptable; some maintenance stays plaintext-shaped on the agent side |

**Tier 0 takeaway:** whole-DB transparent encryption (SQLCipher-style) protects every
row above with **zero logic change** and full functionality, because SQLite sees plaintext
pages only in RAM. **Tier 1 takeaway:** the hybrid retriever's two halves split — vector
similarity becomes homomorphic dot-product (feasible, costly at scale), the FTS keyword
half and plaintext entity-overlap must move client-side or to a leakage-bounded encrypted
index.

## 6. The library decision — OpenFHE via `openfhe-python`

Chosen over a Lattigo (Go) sidecar. Rationale:

- **One process, one language.** The plugin is Python; `openfhe-python` keeps ciphertexts
  in-process — no Go toolchain, no IPC wire protocol for large ciphertext blobs, no second
  process lifecycle to supervise.
- **It exposes exactly what we need.** OpenFHE C++ 1.5.1 (Apr 2026) has Threshold-FHE
  (BGV/BFV/CKKS), interactive bootstrapping for threshold CKKS, **Proxy Re-Encryption**,
  and **CKKS↔FHEW/TFHE scheme switching** (needed for encrypted comparisons / top-k).
  `openfhe-python` (v1.3.1.0) surfaces `Enable(PRE)` (`ReKeyGen`/`ReEncrypt`) and
  `Enable(MULTIPARTY)` (threshold). As of spring 2025 OpenFHE and Lattigo are the only two
  libraries with robust multiparty support; OpenFHE is the only one reachable cleanly from
  Python.
- **Known risk (tracked, not blocking design):** building OpenFHE + `openfhe-python` is a
  non-trivial native CMake/C++ build, and the Python wrapper version lags the C++ lib
  (needs OpenFHE 1.5.1+). **Deployment (confirmed 2026-06-18):** production target is a
  Linux home server (SSH access available); the Windows machine is **dev/test only**. So
  the build happens on Linux, where it is far more tractable — the Windows-build risk is
  effectively retired. **Phase E1 still includes a build/packaging spike** (import +
  generate a CKKS context on the Linux node) before any HE phase commits to the dependency.
  Per principle 3.6 the dep is imported only under Tier 1, so it never touches default/
  Tier-0 users.

Engine choices: **CKKS** for vectors/HRR/similarity/decay (approximate reals; its dot-product
is the single best-fit primitive); **BFV** for exact scalar counts where needed;
**scheme-switch to TFHE** only for the comparison/argmax step of top-k.

## 7. The four open tensions — resolved with the research

**7.1 Runtime decryption — use PRE, not classic threshold (RESOLVED in principle).**
Classic (2,3) Shamir threshold needs ≥2 shares to decrypt; if the agent holds one and the
user is asleep, the agent cannot read its own query results — unacceptable for an
autonomous loop. Resolution: the **runtime path uses Proxy Re-Encryption** — the store
re-encrypts a query's result ciphertext from the storage key to an **agent-only key**,
gated by a query-binding token, so the agent decrypts only results of queries it
initiated. **Threshold/multiparty secret sharing is reserved for the USER audit/reconstruct
path** (god-mode export, key rotation, multi-device). Both are in `openfhe-python`. The
exact PRE protocol + token binding is detailed in Phase E6.

**7.2 Query-binding bounds provenance, not scope (RESOLVED as policy, not crypto).**
"Results of a query I ran" does not limit how *much* a query returns — a compromised agent
could homomorphically "select everything" and decrypt it one query at a time. HE cannot fix
this; it is a policy/threat-model problem. Resolution: **scope limits enforced store-side
and agent-side** — per-cycle query-volume caps, top-k hard ceilings, query-shape
constraints (a recall is always "top-k vs one encrypted probe," never "return all scores"),
and an audit log of re-encryption events the user can review. Documented as an explicit,
conservative, auditable policy (consistent with the anti-fabrication discipline).

**7.3 Top-k is the performance crux (RESOLVED as a phased tradeoff).** A homomorphic
dot-product per stored vector is cheap-ish (768-d packs into one CKKS ciphertext; inner
product = one mult + ~10 rotations); the cost scales linearly with fact count — for hundreds
to low-thousands of facts that's seconds/query, which "cycles-not-seconds" tolerates.
Homomorphic **argmax/sort** over the scores is the expensive part. Two options, sequenced:
(a) **return encrypted scores, rank client-side** — simplest, but leaks the *count* of
facts and costs bandwidth; acceptable for an early spike; (b) **scheme-switch to TFHE for a
homomorphic top-k** — no count leak, much heavier. Phase E3 ships (a); (b) is a later
hardening phase if the leakage profile matters.

**7.4 FTS5 + plaintext entity-overlap die under encryption (RESOLVED by relocation).**
You cannot full-text-search ciphertext, and entity-overlap conflict detection /
clustering need plaintext entity sets. Resolution per principle 3.3: **entity extraction
and any keyword indexing happen client-side at fact creation**, before ciphertext reaches
the store. The store keeps **encrypted entity sets**, and overlap is computed via
**PSI-style encrypted-set intersection**. Keyword recall in Tier 1 either (i) is dropped in
favor of pure homomorphic vector recall, or (ii) uses a **searchable-symmetric-encryption
(SSE) index** with a documented leakage profile. Phase E7 decides (i) vs (ii); the early
HE phases run vector-only recall.

## 8. Phased plan

Each phase: its own test-backed commit, default-OFF/opt-in, with a substrate-validation
query and a named target module, following the established per-phase rhythm
(`baseline green → migration/keystore → store/provider method(s) → config key(s) → tests +
substrate query → inventory/verify_logic update → commit → STOP for review`).

> **E0–E2 are concrete and ready to detail-plan on go-ahead. E3–E7 are sketched; each is
> fully designed only when reached, exactly as the memory roadmap worked.**

**E0 — Encrypted-at-rest substrate (Tier 0). — CODE-COMPLETE + SUBSTRATE-VALIDATED.**
*Goal:* the whole DB opaque at rest under the passphrase, zero functionality loss.

*Result (2026-06-18):* shipped as `crypto_keys.py` + binding selection in `store_common.py`
+ `PRAGMA key` wiring in `store.py` + `encryption_mode`/`encryption_keystore_path` config +
`LatticeMemoryProvider._resolve_encryption_db_key()` wired into `initialize()`. 50 tests
green (3 new: crypto_keys unit, binding selection, subprocess substrate), inventory
regenerated (provider 36→37), verify_logic PASS. The env signal `RESONANT_LATTICE_DB_ENCRYPTED`
selects the SQLCipher binding at import; `RESONANT_LATTICE_PASSPHRASE` (or setup) supplies the
passphrase; the keystore is auto-created on first run with a no-recovery warning. *Follow-up
polish:* interactive setup prompt + recovery export (hermes-framework integration), and a note
that encrypted-DB embedding-dim detection relies on the Ollama probe (the plaintext
`_read_db_vector_dim` read returns None on an encrypted file and is silently bypassed).

*Spike — PASSED 2026-06-18 (Windows Py3.13, throwaway):* `sqlcipher3-wheels==0.5.7`
(SQLite 3.51.1) loads the `sqlite_vec` (0.1.9) loadable extension, runs FTS5, enforces
`content UNIQUE`, writes an opaque file (random-salt header, not `SQLite format 3`), rejects
the wrong key, and reopens with the right key — all 9 checks green against a vec0+fts5+trigger
schema mirroring the real store. **Decision locked: SQLCipher whole-DB encryption (Option A).
The app-layer AEAD fallback is RETIRED** (it would have sacrificed server-side FTS). Linux
(production target) gets sqlcipher3 wheels too, with build-from-source as backstop.

*Approach:* SQLCipher transparent whole-DB encryption — FTS5/sqlite-vec/UNIQUE stay intact
because pages decrypt only in RAM. We pass a **raw 32-byte key** (`PRAGMA key = "x'…'"`),
doing our own Argon2id rather than SQLCipher's weaker PBKDF2.

*Implementation steps (locked):*
1. `crypto_keys.py` (new flat leaf) — `passphrase → Argon2id → master → HKDF("lattice-rest-v1")
   → 32-byte raw DB key`; keystore sidecar (salt/params/version/key-check, asserted
   secret-free); pluggable key-source (env/explicit/prompt; sealed sources deferred to E1);
   best-effort mlock/zeroize. HKDF is stdlib-hmac (no `cryptography` dep).
2. `store_common.py` — extend the existing centralized binding selection to pick `sqlcipher3`
   when encryption is on (preserves the shared `IntegrityError` identity). Dedicated test.
3. `store.py` — `PRAGMA key` as the first statement on the encrypted connection; rest unchanged.
4. `config_schema.py` — `encryption_mode` (`none` default | `at_rest` | `blind` reserved) +
   `encryption_keystore_path`.
5. Provider/setup wiring — fetch the key from the configured source at init; `hermes memory
   setup` passphrase prompt + destructive/irrecoverable warning + optional recovery export.
6. Tests + harness — encrypted round-trip, wrong-key failure, binding test, parametrized run
   of the existing suite under encryption; update `before_inventory.json` + verify_logic
   EXPECTED_DIFFS. Also: `encrypt_existing_db` / `decrypt_to_plaintext` via `sqlcipher_export`.

*New deps (only when `encryption_mode != none`):* `argon2-cffi`, `sqlcipher3-wheels`.
*Substrate validation:* file header is not `SQLite format 3` and the plain `sqlite3` CLI
fails to open it; with the right passphrase the full existing suite passes under
`encryption_mode=at_rest`; wrong key fails cleanly.

**E1 — Key hierarchy & setup + HE build spike. — BUILD SPIKE PASSED.** *Goal:* extend the
master→HKDF hierarchy (from E0 `crypto_keys.py`) to the HE keypair + PRE rekey + agent key,
plus confirm OpenFHE works on the target node.

*Spike — PASSED 2026-06-18 on the Linux server (Ubuntu 24.04.4, 12c/15GB):* **no source
build required.** The `openfhe` PyPI wheel `1.5.1.0.24.4` (4 MB, per-Ubuntu-version tagged —
`.24.4`=24.04) installs into a venv and functionally validates all three primitives we need:
**CKKS** (encrypt → homomorphic add `[6,8,10,12]` + mult `[5,12,21,32]` → decrypt), **PRE**
(encrypt under key A → re-encrypt → key B decrypts `[11,22,33,44]` while A's key yields garbage
— the runtime agent-use-key model, with the security property confirmed by the negative
control), and **Multiparty** (joint `MultipartyKeyGen`). The whole OpenFHE build risk is
retired. Install recipe: `python3 -m venv ~/he && ~/he/bin/pip install openfhe`.

*Environment facts (locked):* **no TPM** on the node (`/dev/tpm*` absent) → key-at-boot uses
the **SSH-unlock-once-per-boot** model (doc §4.2 fallback), not TPM2. Pre-existing system
snag (NOT ours, doesn't block): a pending `linux-image-6.8.0-124` update fails its
`nvidia-fs` DKMS post-install, so `apt` errors until the user resolves it. (`nomic-embed-text:latest`
and `ibm/granite4.1:8b` are both present on the node — confirmed.)

*Remaining E1 (when resumed):* generate the HE keypair deterministically from the master,
persist only public/eval keys + the `rk_master→agent` rekey in the keystore, unit-test that
no secret key / master touches disk. *Config:* setup-flow fields in `CONFIG_SCHEMA`. *Target:*
`crypto_keys.py` (HE extension), setup wiring in the provider lifecycle.

**E2 — CKKS encrypted vector similarity (blind recall). — ✅ COMPLETE (core + all plumbing,
2026-06-18).** *Goal:* the single most important HE proof: store encrypted embeddings, compute
recall similarity homomorphically, never decrypt store-side. *Result:* the make-or-break HE risk
is RETIRED — real CKKS blind recall ranks identically to the plaintext `LatticeRetriever` on the
node (`test_store_he_blind_vs_plaintext_topk`: N=10, dim=768, blind top-k == plaintext top-k ==
numpy truth, decrypted-cosine max error 5.0e-12), and the store side cannot decrypt
(`SecretRequiredError`, E2 core).

*Core — DONE 2026-06-18 (`he_crypto.py`, commit f92b826):* `BlindCrypto` validated 3-process
on the node — setup keygen; a fresh eval-only store scores 10 facts homomorphically and
**cannot decrypt** (`SecretRequiredError`); the client decrypts with HE cosine == plaintext
cosine (max err 0.0) and identical top-k. User approved the 3 decisions (client-side ranking /
count-leak OK for E2; random keygen + AES-GCM-wrap the secret under the master; standalone
proof, not provider-wired). CKKS depth 1 / batch 1024 / ~787 KB per ct.

*Plumbing remaining (lower-risk):* (1) ✅ DONE 2026-06-18 — `crypto_keys` AES-256-GCM wrap/unwrap
of the HE secret under an HKDF master subkey (`derive_he_wrap_key` + `wrap_he_secret` /
`unwrap_he_secret`; finishes E1's key hierarchy; adds the `cryptography` dep, blind-tier-gated &
lazy-guarded; 51 tests green, inventory/verify_logic clean); (2) ✅ DONE 2026-06-18 (store-side) —
`semantic_he(id, ct BLOB, he_version)` table migration (CASCADE-FK to facts) + `store_blind.BlindMixin`
(`store_he_vector`/`get_he_vector`/`iter_he_vectors`/`count_he_vectors`; pure opaque-blob ops, no
openfhe on the store side, 52 tests green); the client-side encrypt-then-write orchestration folds
into (3); (3) ✅ DONE 2026-06-18 — `BlindRetriever(LatticeRetriever)` in `retrieval.py`
(`blind_search`/`blind_search_vec`/`blind_scores`/`_materialize_blind`: encrypt query → scan
stored cts → homomorphic score → decrypt → top-k; duck-typed on a `he_crypto.BlindCrypto` client,
so no HE import in retrieval; vector-only, client-side rank, superseded-filtered + min_similarity
floor for parity). Orchestration validated on Windows via a plaintext stand-in crypto (53 tests
green); real CKKS correctness is (4) on the node. (4) ✅ DONE 2026-06-18 — comparison test
`test_store_he_blind_vs_plaintext_topk` (real CKKS, N=10/dim=768, strictly-separated fixture
cosines): blind top-k == plaintext `LatticeRetriever` top-k == numpy truth, decrypted cosine within
5.0e-12 of the construction targets; run for real in the node's `~/he` venv (self-skips without
openfhe, so the Windows suite stays green at 54). *Config:* gated under `encryption_mode=blind`.
*Substrate validation:* `semantic_he` rows hold only CKKS ciphertext (no float recoverable); blind
recall matches the plaintext retriever's top-k. *Dev workflow:* author on the Windows repo → USER
syncs `resonant_lattice\*.py` → `<remote>/resonant_lattice/` via WinSCP (agent-side `scp`
is blocked by the data-exfil guardrail — the LAN node reads as untrusted-external), then Claude runs
the node test over SSH (command-exec is allowed). NOTE the synced path is the `resonant_lattice/`
SUBfolder (WinSCP preserves the tree), so run from there.

**E3 — Homomorphic top-k hardening. — CORE PROVEN (blind argmax), 2026-06-18.** Replace
client-side ranking with TFHE scheme-switched top-k to remove the count leak (tension 7.3 option b).

*Spike — PASSED on the node:* OpenFHE-python exposes the full scheme-switching suite
(`EvalSchemeSwitchingSetup/KeyGen`, `EvalCKKStoFHEW`/`EvalFHEWtoCKKS`, `EvalMaxSchemeSwitching`
with `SchSwchParams.SetComputeArgmin/SetOneHotEncoding`). The full pipeline was validated
end-to-end: encrypt query + N fact vectors → pack per-fact `EvalInnerProduct` results into one
score vector → `EvalMaxSchemeSwitching` → one-hot argmax == plaintext argmax (N=8, dim=16, ~2.7 s).
**Key property:** the argmax takes the PUBLIC key, not the secret, so the store learns only the
winning index — neither the scores nor their ranking leak. Two scaling facts locked: scores must be
scaled into the FHEW-safe range (cosines×0.5) or large values wrap; depth = 2 (packing) + 13 +
ceil(log2 N).

*Shipped (`he_crypto.BlindArgmax`, `test_he_blind_argmax_pipeline`):* the proven recipe as a
single-context engine — `generate(dim, num_facts)` / `encrypt_vector` / `argmax` (store-side,
public+eval only) / `decrypt_onehot` (client). Inert without openfhe (Windows suite green).

*0a finding (2026-06-19) — the FHEW path canNOT be a blind (split) mode:* the FHEW/scheme-switching
keys do not serialize in openfhe-python 1.5.1 (latest wheel), so a deserialized eval-only store
segfaults in `EvalMaxSchemeSwitching`. **E3's blind path therefore pivots to pure-CKKS polynomial
comparison** (`he_crypto.BlindArgmaxCKKS`: Chebyshev `abs`/`exp`, Cheon et al.) — only CKKS
mult+rotation keys, which serialize, so the untrusted store CAN run it (node-proven). The FHEW
`BlindArgmax` remains the faster *co-located-only* reference.

*Remaining E3 plumbing (on the pure-CKKS engine):* pack query/fact inner products into the comparison
context for full blind top-k recall (the score-vector argmax is proven; integration is 0c-adjacent —
node-spike first); top-k>1 (depth budget → bootstrapping between rounds; bootstrap keys serialize, so
the split holds); non-power-of-two fact counts via the `_CMP_PAD` sentinel (proven for the padded
case) / −∞ padding; a production security level + measured latency (the proof uses `HEStd_NotSet`,
§3b). Gated; pursued when the leakage profile warrants the cost.

**E4 — HRR operations under HE.** Encrypted `bind`/`similarity` (phase-add/dot-product —
CKKS-native) for relational recall (P5) and conflict similarity on the blind store. Confront
`bundle`/`mod 2π`/`cos` non-polynomial ops (polynomial approximation or move client-side).

**E5 — Dream-cycle maintenance on ciphertext.** Homomorphic decay (scalar multiply),
encrypted conflict detection (encrypted-set overlap + CKKS similarity band), promotion/
eviction on encrypted resonance (BFV comparisons or scheme-switch). The maintenance loop
must keep running blind — this is where "cycles-not-seconds" pays off hardest.

**E6 — PRE runtime path + threshold user-audit path. — CORE PROVEN, 2026-06-18.** Implement
tension 7.1: store-assisted proxy re-encryption of query results to an agent-only key with
query-token binding + scope limits (7.2); threshold/multiparty reconstruction for the user's
god-mode audit/export/rotation.

*Spike — PASSED on the node (the full three-key model of §4.1):* **PRE** — generate a storage
(master) key + an agent use-key + `rk_storage->agent`; encrypt a result under the storage key;
the store `ReEncrypt`s it to the agent; the agent decrypts the re-encrypted result `[0.42,0.10]`
but the agent key on the RAW storage ciphertext is **REJECTED** (OpenFHE refuses the decode) —
the "use but can't read" property, impossible in Tier 0. Master = god-mode. **Threshold** —
2-of-2 multiparty (`MultipartyKeyGen` + `MultipartyDecryptLead`/`Main`/`Fusion`): both shares
fused reconstruct `[0.77,0.33]`; a single share is **REJECTED**. So no single party (store or
agent) decrypts alone; the user holding the shares reconstructs anything.

*Shipped:* `he_crypto.BlindPRE` (keygen / rekey / encrypt / reencrypt / decrypt) and
`he_crypto.ThresholdAudit` (first_party / join / encrypt / partial_lead / partial_main / fuse),
both inert without openfhe; `blind_policy.ScopeLimiter` + `ReEncryptAuditLog` (PURE-Python, §7.2):
top-k ceiling + per-cycle query cap + per-cycle re-encryption cap (logical cycle, never
wall-clock) with a user-reviewable audit log — the policy bound on the honest seam (HE bounds
provenance, not scope). Tests: `test_he_pre_and_threshold_audit` (self-skips w/o openfhe, proven
on node), `test_blind_policy_scope_limiter` (runs everywhere). `crypto_keys.wrap_he_secret`
already wraps the agent use-key blob under the master (it is AEAD over any bytes).

*Remaining E6 plumbing:* serialized key-blob split across real store/agent/user processes;
binding the query token end-to-end (ScopeLimiter token -> the store's ReEncrypt gate); unifying
the PRE context with the E2 scoring context so actual cosine-score cts are what gets re-encrypted;
(t,n) beyond 2-of-2; provider/setup wiring. Crypto core is fully de-risked.

**E7 — Encrypted keyword recall & entity overlap.** Decide FTS drop vs SSE index (tension
7.4); implement client-side entity extraction → encrypted entity sets → PSI overlap for
conflict detection/clustering on the blind store.

## 9. Performance & hardware budget (Principle 2 guardrail)

The risk vs "minimal hardware (RTX 3090 Ti)": CKKS contexts are RAM-hungry and bootstrapping
is ~20s/op on CPU (2025 work shaves 20–40%). Mitigations: keep recall in the leveled regime
(avoid bootstrapping on the hot recall path; reserve it for deep maintenance circuits);
linear-scan dot-products are embarrassingly parallel; cap fact count per the existing
`max_long_facts`; run the heaviest circuits on the dream cycle, not on recall. Every HE
phase reports measured per-op latency on the target node as part of acceptance.

## 10. Validation approach (substrate, per working style)

Acceptance is always a **row-level DB / ciphertext check**, never "ask the agent what it
remembers." For Tier 0: prove the file is opaque without the key and identical-behaviour
with it. For Tier 1: prove store-side rows are non-decryptable with only the eval key, that
blind recall matches plaintext recall on fixtures, and that the re-encryption audit log
records every runtime decrypt. The existing suite (`test_resonant_lattice.py` plus the
end-to-end `tests/live_e2e.py`) extends per phase.

## 11. Open decisions still needed before the corresponding phases

1. **Passphrase entry at process start (E0/E1). — RESOLVED 2026-06-18, see §4.1–4.2.**
   Three-key PRE model: passphrase only at setup (then destroyed); unattended boot uses a
   sealed agent use-key (TPM2 preferred, once-per-boot SSH unlock fallback). The use-key
   reads only re-encrypted query results, not the raw DB.
2. **Tier-0 mechanism if SQLCipher is painful on Windows (E0).** Accept the SSE-less
   app-layer-AEAD degraded mode (lose server-side FTS), or require the store on Linux?
3. **Tier-1 keyword recall (E7).** Drop FTS entirely (pure vector recall) or build an SSE
   index with a documented leakage profile?
4. **Top-k leakage tolerance (E2 vs E3).** Is leaking the *fact count* acceptable for the
   early blind-recall spike, or must we go straight to TFHE top-k?
5. **Agent autonomy vs PRE availability (E6).** When the user (and thus the master secret)
   is absent, exactly what is the agent permitted to decrypt, and what is the per-cycle
   scope cap?

## 12. Dependency graph

```
E0 At-rest ──► E1 Keys+HE-spike ──► E2 Blind vector recall ──┬─► E3 HE top-k
                                                             ├─► E4 HRR under HE
                                                             ├─► E5 Dream-cycle blind
                                                             ├─► E6 PRE runtime + threshold audit
                                                             └─► E7 Encrypted keyword + entity PSI
```

E0 and E1 are independent of HE correctness and deliver the user's literal goal early. E2 is
the make-or-break HE spike. E3–E7 harden and complete the blind store and are each designed
in full when reached.

## 13. Deployment prerequisites (Linux production server)

Production target = a Linux home server running hermes-agent (SSH available); the Windows
machine is **dev/test only**. E0/E1 *logic* is built and tested on Windows; the Linux box is
needed from **E1's OpenFHE build spike** onward. Staged setup (Debian/Ubuntu shown — adjust
package names for other distros):

- **Runtime baseline (any tier):** Python 3.11/3.12; Ollama (local preferred; fast cloud like deepseek-v4-flash:cloud works great for memory layer)
  and `ollama pull nomic-embed-text`; pip `numpy`, `sqlite-vec`. Verify the stdlib `sqlite3`
  allows extension loading (`python3 -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)"`);
  if it errors, install `pysqlite3-binary` (the plugin already falls back to it).
- **Tier 0 (encrypted-at-rest):** `pip install argon2-cffi sqlcipher3-wheels`. (If no
  Linux wheel for the target Python, build sqlcipher3 from source with extension loading
  enabled — straightforward on Linux.)
- **Tier 1 / E1 OpenFHE build spike:** `sudo apt install -y build-essential cmake git
  python3-dev python3-pip libomp-dev autoconf libtool`; ≥8 GB RAM free + a few GB disk for
  the compile. The actual OpenFHE + `openfhe-python` build runs over SSH during E1.
- **Tier 1 pip deps:** the `~/he` venv needs `openfhe` (Linux wheel) + `cryptography` (the
  AES-256-GCM wrap of the HE secret, E2.1). Both confirmed present in the `~/he` venv 2026-06-18
  (alongside `numpy`, `sqlite_vec`, `argon2-cffi`). NOTE: `sqlcipher3` is installed for `ccode`
  in `~/.local` (user site, a working py3.12 wheel) but NOT in the isolated `~/he` venv — run
  `~/he/bin/pip install sqlcipher3-wheels` there if Tier-0+Tier-1 composition is wanted (optional
  for E2; the blind path and the comparison test run on a plain SQLite file).
- **Key-at-boot (E1/E6, check only for now):** `ls /dev/tpm*` — if `/dev/tpm0` or
  `/dev/tpmrm0` exists, we prefer TPM2-sealing (`tpm2-tools`) for the agent use-key;
  otherwise the once-per-boot SSH-unlock fallback (no extra setup).
- **Access:** an SSH user account with the above toolchain + git, set up when E1 begins.

## 14. Remaining-work execution plan (next-session guide)

**Status recap (2026-06-19, HEAD `c670322` on `encryption-northstar`).** Priorities 0–5 (the numbered
phase CORES E0–E7) are DONE + node-validated; `encryption_mode=blind` is a real switch-on-able tier: **0a**
engines + FHEW→pure-CKKS pivot (`d96aa99`); **0b/0c** keystore + `BlindWriter` + blind write/recall wiring
(`5d39f45`/`2c9b7db`); **E3 §3b** 128-bit (`3e85ed0`); **E6 6c** token-gated re-encryption + audit
(`9fed109`); **E4 4a/4b** HRR via the lift (`45566ed`/`d7337cf`); **E5 5a/5b** decay + threshold compare
(`494686e`/`0de065c`); **E7 7b** AEAD entity sets (`77c0661`).

**NOW: Priority 6 — PROVIDER WIRING (route the blind helpers into the live provider).** Done so far:
**P1** blind entity-set write mirror (`0b4fecc`); **2a** multi-keyset HE keystore (Option A — one keystore
holds recall@embed-dim + HRR@2·hrr-dim + a LIGHT decay-only maint keyset; engines serialize eval keys
by-tag so the keysets coexist in one process) (`e179bc1`, node-val); **2b-i** multi-context `initialize`
switch + HRR write mirror to `semantic_he_hrr` (`c4d074a`); **2b-ii(a)** blind HRR recall uses the separate
2·hrr_dim client (`d378488`, node-val). So a CONSOLIDATION-created fact mirrors its embedding + entity set +
HRR lift into the encrypted tables on write. **Locked design facts:** Option A keysets coexist (OpenFHE keys
the global eval-key store by context tag); maint is generated at depth 1 (~0.8MB) and decays FROM ORIGIN
(one mult by `factor**elapsed`, unbounded cycles) — NOT in-place compounding (survives only `depth` cycles);
`semantic_he_hrr` holds the FACT-CONTENT HRR (≠ `relational_recall`'s per-triple HRR).

**Priority 6 — remaining (recommended order):**
- **6a. Write-path completeness — DONE (`c670322`), node-validated.** A single idempotent provider
  `_blind_reconcile()` pass finds `semantic_facts` rows lacking blind ciphertext (`store_blind.
  facts_missing_blind`, a LEFT JOIN, superseded-excluded, `limit`-batched) and mirrors embedding
  (`store_facts.get_fact_embedding` — exact float32 read-back from `semantic_vec`, NO Ollama) + entities
  (`get_entities_for_fact`) + HRR (`get_fact_hrr_phases`→`hrr_lift`) into the three blind tables. Called
  at the END of the consolidation epoch + dream cycle (catches abstraction/gist/procedural store-side facts
  + builtin) and incrementally backfills a first-blind-enable store (`blind_reconcile_batch`, default 200,
  spreads across cycles). REPLACED the per-fact eager hooks (0b/P1/2b-i) with this one mechanism. Node test
  `test_blind_reconcile_backfill` (backfill 3 tables from plaintext → blind recall == plaintext) +
  `test_blind_reconcile_readback_helpers` (Windows). Edge cases noted: HRR re-encode migration ⇒ stale
  `semantic_he_hrr` (clear+re-mirror — follow-up); embed-model change ⇒ re-embed (rare migration).
- **6b. 2b-ii(b) resonance + dream-cycle blind maintenance (NEXT).** DEFERRED pending the §5 source-of-truth
  decision (blind decay is redundant with the plaintext dream cycle until §5 makes encrypted resonance
  authoritative). Needs the decay-from-origin `BlindMaintainer` + a `set_cycle` column on `semantic_he_meta`.
- **6c. Blind recall routing into the conflict path** (content-HRR `blind_hrr_search` + `BlindEntityStore.
  find_conflicts`, client-side) — entangled (the store-side conflict pass is plaintext today).
- **6d. 6e PRE-runtime recall + `memory_audit` over `reencrypt_audit`** (not §5-blocked).
- **6e. at_rest+blind compose; interactive `hermes memory setup`.**
Horizon = the full §5 blind store (AEAD content/source; encrypted resonance becomes source of truth).
**Validation seam:** helper layer node-validated at the substrate; provider glue harness-validated only
(Hermes isn't installable on the dev box/node) — every wired component is node-proven.

**Working discipline (unchanged, applies to every item):** spike any *unproven* crypto on the node
first via an SSH heredoc (no file sync needed — it only needs `openfhe`); then implement on the
Windows repo; then validate the *committed* code on the node after a **WinSCP re-sync**
(`resonant_lattice\*.py` → `<remote>/resonant_lattice/`, run from that subfolder —
agent-side `scp` is blocked by the data-exfil guardrail). Keep the suite green
(`python resonant_lattice/test_resonant_lattice.py`), commit per phase, default-OFF/gated.
**Locked recipes to reuse:** E2 CKKS (depth1/scale50/batch=next_pow2(dim)); E3 argmax
(SCHEMESWITCH, TOY FHEW, `SetComputeArgmin`+`SetOneHotEncoding`, scores×0.5 into FHEW-safe range,
depth = 2 + 13 + ⌈log₂N⌉, `EvalCKKStoFHEWPrecompute` required before the max); E6 PRE
(`SetPREMode(INDCPA)`, `ReKeyGen`/`ReEncrypt`) + multiparty (`MultipartyKeyGen`/`DecryptLead`/
`Main`/`Fusion`).

### Priority 0 — Integration foundation (do FIRST; unblocks everything below)
The biggest gap is that the blind tier is proven but not real. Three sub-pieces:
- **0a. Unified-context decision + serialized-key-split spike (node, make-or-break). — DONE +
  node-validated 2026-06-19; engines refactored.** Result of the 3-process serialized-split spikes on
  the node (openfhe-python 1.5.1.0.24.4, the LATEST wheel):

  | Path | Splits across an untrusted store? | Evidence |
  |---|---|---|
  | E2 cosine + E6 PRE (one `{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,PRE}` ctx, 128-bit) | ✅ yes | store scores **and** `ReEncrypt`s the score ct to the agent with NO secret; agent decrypts (err ~3e-5), agent key REJECTED on the raw ct, master = god-mode |
  | Multiparty audit (`{…,MULTIPARTY}`) | ✅ yes | serialized shares fuse; a single share REJECTED |
  | E3 FHEW scheme-switch argmax | ❌ **no** | FHEW `lwesk`/`BinFHEContext` have no `Serialize`; a deserialized eval-only ctx **segfaults** in `EvalMaxSchemeSwitching`; `EvalSchemeSwitchingKeyGen` needs the secret. 1.5.1 is the latest wheel — no upstream fix |
  | 🆕 pure-CKKS comparison argmax (Chebyshev `abs`/`exp`, Cheon et al.) | ✅ yes | only CKKS mult+rotation keys (which serialize); eval-only store runs full argmax → client one-hot == plaintext argmax |

  **Decision (user-confirmed): purpose-built contexts, NOT one monolith** — a unified `Enable` of all
  features works but buys nothing for the binding serialization constraint and forces E3's heavy params
  on cheap ops. **E3 pivots from FHEW scheme-switching to pure-CKKS comparison** for any *blind*
  (untrusted-store) argmax; the FHEW `BlindArgmax` stays as a faster *co-located-only* reference.
  Engines refactored in `he_crypto.py`: **`BlindRecallPRE`** (unified cosine+PRE, serializable, 3 roles
  `generate`/`load_eval`/`load_client`/`load_user`; folds E6 item 6b — the actual cosine score ct is
  what gets re-encrypted) and **`BlindArgmaxCKKS`** (pure-CKKS comparison argmax, serializable). Two
  self-skipping tests added (`test_he_recall_pre_split`, `test_he_argmax_ckks_pipeline`); Windows
  harness green (59 tests, inventory/verify_logic clean — `he_crypto.py` is untracked). *Note:* score-
  vector argmax is proven; packing query/fact inner products into the comparison context (full blind
  top-k recall) is the 0c / E3-hardening integration step (node-spike first).
- **0b. Blind WRITE path wiring. — DONE (commits `5d39f45` + `2c9b7db`), node-validated.** New
  `crypto_keys` HE keystore (`create/save/load_he_keystore`, `he_keystore_is_secret_free`,
  `setup_or_load_blind_client` — generate+wrap+persist on first run / load+unwrap later, role
  user|agent) + `retrieval.BlindWriter` (client-side encrypt → `store.store_he_vector`). Provider
  `_resolve_blind_client()` + the `_run_consolidation_epoch` hook mirror each added fact's embedding
  into `semantic_he`. *Substrate (node):* `semantic_he` fills with opaque CKKS cts (plaintext not
  recoverable); keystore is secret-free + the master round-trips; 2-process keystore load works.
- **0c. Blind RECALL wiring. — DONE (commit `2c9b7db`), node-validated.** `initialize()` swaps in
  `BlindRetriever` (backed by the keystore-loaded `BlindRecallPRE` client, secret unwrapped via
  `crypto_keys`) under `encryption_mode=blind`; `BlindRetriever.search` override routes the existing
  recall path (`_compute_prefetch`) through blind recall. *Acceptance (node):* end-to-end
  keystore→BlindWriter→BlindRetriever recall ranks identically to plaintext (max cosine err 7e-13).
  *Validation note:* the provider class's blind branch is gated/defensive and harness-validated
  (py_compile + inventory + verify_logic + stub_loader import/construct); every component it wires is
  node-proven (Hermes isn't installable on the dev box or node, so the class itself isn't run live).
  *Scope:* hooks the main consolidation ingest; lifecycle builtin-mirror + abstraction write paths,
  and at_rest composition, are follow-ups.

### Priority 1 — E3 hardening
- **3a.** Serialized scheme-switching key split (depends on 0a).
- **3b. Production security level:** replace `HEStd_NotSet` + `TOY` FHEW with `HEStd_128_classic` +
  `STD128`; re-tune ring/depth; **measure per-op latency on the node** (acceptance reports it, §9).
- **3c.** Non-power-of-two fact counts: −∞ padding for empty argmax slots so padding never wins.
- **3d. top-k>1:** each `EvalMaxSchemeSwitching` exhausts the depth budget → needs inter-round
  re-leveling. Evaluate (spike first): (i) iterative argmax + bootstrapping between rounds (heavy —
  measure), (ii) ship blind k=1 + fall back to E2 client-side ranking for k>1, (iii) batched variants.

### Priority 2 — E6 wiring (the runtime trust model, made real)
- **6a. — DONE (0a, `d96aa99`).** Serialized key split is the `BlindRecallPRE` 3-role model: storage
  eval keys + rk (store, `load_eval`) / agent use-key (agent, `load_client`) / master (user, `load_user`).
  Keystore (`crypto_keys.setup_or_load_blind_client`) persists the public/eval/rk blobs + the AES-GCM-
  wrapped master + agent secrets. Node-proven 4-process split.
- **6b. — DONE (0a, `d96aa99`).** PRE is unified with the E2 scoring context inside `BlindRecallPRE`:
  `cosine_score` then `reencrypt_score` operate on the SAME context, so the actual cosine-score ct is
  what gets re-encrypted to the agent (no separate `BlindPRE` context). Node-proven (agent err ~4e-13).
- **6c. — DONE (2026-06-19), validated.** `blind_policy.BlindReEncryptGate` (store-side single-use
  token gate: `register`/`spend`/`remaining`, refuses over-spend / unknown / replay) binds each
  `ReEncrypt` to one `ScopeLimiter.authorize` token; the audit log is PERSISTED in a new `reencrypt_audit`
  table (`store_schema._migrate_add_reencrypt_audit` + `store_blind.record/get/count_reencrypt_events`).
  *Substrate:* `SELECT cycle, query_token, k FROM reencrypt_audit`. *Node (real CKKS):* eval-only store
  scores + token-gated `reencrypt_score` → AGENT (not master) decrypts → correct ranking; over-budget +
  replay refused; grant persisted.
- **6d.** (t,n) beyond 2-of-2 for the user-audit path; multi-device share distribution. *(pending)*
- **6e. — PARTIAL.** Store-side surfacing done (`get_reencrypt_events` for review / memory_audit); the
  PROVIDER opt-in PRE-runtime recall (config flag + agent-role recall using the gate) and a dedicated
  `memory_audit` action over the log are the remaining provider glue. *(pending)*

### Priority 3 — E4 HRR under HE (spike-first)
- **4a. — DONE (2026-06-19), node-proven.** The clean reduction: HRR `similarity(a,b) = mean(cos(a−b))
  = (1/dim)·Σ[cos a·cos b + sin a·sin b] = dot(lift(a), lift(b))` where `lift(φ) = (cos φ, sin φ)/√dim`
  is **already L2-unit**. So **encrypted HRR similarity needs NO new crypto — it is the E2 inner product
  over the client-side lift** (`holographic.hrr_lift`; HE dim = 2·hrr_dim). Node-validated vs
  `holographic.similarity` via the committed `BlindRecallPRE.cosine_score` (err ~6e-13; related rephrase
  0.58 ≫ unrelated 0.05; unbind round-trip 1.0). Non-polynomial ops RESOLVED by relocation, not
  approximation: `cos`/`sin` computed client-side in the lift, `mod 2π` unnecessary (the lift encodes the
  angle; cos periodicity), `bundle` stays client-side at encode (principle 3.3). Encrypted **bind** also
  proven as the elementwise complex-multiply under HE (`cos_c = ca·cb − sa·sb`, `sin_c = sa·cb + ca·sb`;
  err ~1e-12) for the rare case the store must compose blind — but recall does not need it (binding is
  client-side at encode). *Shipped:* `holographic.hrr_lift` + tests `test_hrr_lift_identity` (everywhere)
  / `test_he_hrr_similarity_via_lift` (node).
- **4b. — CORE DONE (2026-06-19), node-validated.** The blind HRR store/recall path: a `semantic_he_hrr`
  table (`store_schema._migrate_add_semantic_he_hrr`, mirror of semantic_he, CASCADE-FK); the blind-vector
  methods (`store_blind.store/get/iter/count_he_vectors`) generalized with an allowlisted `table` selector;
  `retrieval.BlindWriter(table=…)` + `BlindRetriever.blind_hrr_scores`/`blind_hrr_search` (lift the phase
  probe client-side via `hrr_lift`, encrypt, homomorphic cosine over the stored encrypted lifts). *Substrate:*
  `semantic_he_hrr` fills independently of `semantic_he`, allowlist-guarded, CASCADE-cleaned. *Node (real CKKS):*
  blind HRR recall ranking == plaintext `holographic.similarity`, err ~6e-13. **Deferred (provider wiring,
  entangled with E5/E7):** mirroring fact HRR into `semantic_he_hrr` on write + routing `relational_recall`'s
  fuzzy pass / conflict similarity through the blind path under `encryption_mode=blind` (the exact-match graph
  pass needs E7 encrypted entity-matching to be fully blind); the blind client must also be sized for the
  larger of embed-dim vs 2·hrr_dim (or a second HRR-dim client).

### Priority 4 — E5 dream-cycle on ciphertext (the philosophical payoff)
- **5a. — CORE DONE (2026-06-19), node-validated.** `he_crypto.BlindMaintenance` (pure-CKKS,
  serializable; `generate`/`load_eval`/`load_client`): homomorphic **decay** (`resonance *= factor`,
  scalar mult, exact, ~4ms) + threshold **comparison** (`step(resonance − threshold)` -> encrypted 0/1
  indicator for promotion/eviction). KEY CHOICE: the comparison uses the SAME pure-CKKS Chebyshev
  sign-approx as `BlindArgmaxCKKS` — **NOT BFV/scheme-switch** (scheme-switch doesn't serialize, 0a) — so
  it splits across store/client and the store runs maintenance with public+eval keys only, no secret.
  Resonance scaled to ~[0,1] client-side; classification exact outside a small transition band (the
  exact-boundary case decrypts to ~0.5 — fine for a soft every-cycle signal). Node-measured: ~3.5s/compare
  at HEStd_128_classic — lands on the dream cycle, not the hot path (§9). Conflict similarity bands =
  `ge_threshold` on a score ct (low/high); the entity-overlap half of conflict detection is E7 (PSI). Test
  `test_he_blind_maintenance`.
- **5b. — CORE DONE (2026-06-19, user sign-off + node-validated).** DECISION: **decay fully blind,
  promotion/eviction CLIENT-ASSISTED** (fully-autonomous store-side tier evolution under HE is hard/lossy
  and was declined). Encrypted resonance scalar now persists in `semantic_he_meta`
  (`store_schema._migrate_add_semantic_he_meta`; resonance/max scaled to ~[0,1], CASCADE-FK).
  `he_crypto.BlindMaintenance` gained ct (de)serialization (`serialize_ct` + blob-accepting decay/compare/
  decrypt) for storage. `retrieval.BlindMaintainer`: `set_resonance` (encrypt+store), `decay_all(factor)`
  (STORE decays every encrypted resonance and writes it back, NEVER reading it — fully blind), `get_resonance`
  / `settle(promote_thr, prune_thr)` (CLIENT decrypts + thresholds in plaintext → promote/evict fact lists
  for the caller to apply via the existing plaintext `promote_facts`/`remove_fact`). *Node:* decay blind +
  settle == plaintext (promote 2 / evict 1). Tests `test_store_he_meta_table_substrate`/`test_he_blind_maintainer`.
  **Deferred (provider wiring):** route the dream cycle to `decay_all` each cycle + `settle` on a client visit
  under `encryption_mode=blind`; the `ge_threshold` homomorphic flag stays available for autonomous candidate-
  flagging + conflict-similarity bands. The full §5 blind store (encrypt content/entities too) remains the horizon.

### Priority 5 — E7 encrypted keyword + entity overlap
- **7a. DECIDED (2026-06-19, user sign-off): DROP FTS in blind mode — pure vector recall.** Blind recall
  is embedding + HRR cosine only; the FTS5 index is not maintained on the blind store (~zero keyword
  leakage, matches "the store learns nothing"). Accepted cost: exact keyword matching for poorly-embedding
  technical names/IDs degrades to embedding rank; exact lookups go via `get_fact <ID>`. The SSE option was
  declined (its search/access-pattern leakage is hard to bound against a persistent adversarial store).
  *Implementation when E7 is built:* `BlindRetriever` already runs vector-only (no FTS half) — so 7a is
  largely a no-op confirmation; just ensure blind mode never builds/relies on `semantic_fts`.
- **7b. — CORE DONE (2026-06-19, user sign-off: CLIENT-SIDE, no-leak), substrate-validated.** Per-fact
  entity NAME sets are AEAD-encrypted at rest in `semantic_he_entities` (`crypto_keys.encrypt_entities`/
  `decrypt_entities` under a master-derived `derive_entity_key`; RANDOM GCM nonce so identical sets are
  indistinguishable on disk → the store learns NO entity co-occurrence). Overlap / conflict-candidate
  finding run CLIENT-side on the decrypted sets (`retrieval.BlindEntityStore.set_entities`/`get_entities`/
  `overlap`/`find_conflicts`) — the store-side deterministic-token PSI was DECLINED (same leakage class as
  the SSE drop). `store_blind` table allowlist + `_migrate_add_semantic_he_entities` (mirror semantic_he,
  CASCADE). Pure AEAD + SQLite (no openfhe) → substrate-validated on Windows (opaque blobs, plaintext not
  recoverable, client overlap/find_conflicts == plaintext, CASCADE). Tests `test_crypto_entity_aead`/
  `test_store_blind_entities`. **Deferred (provider wiring):** mirror each fact's entities into
  `semantic_he_entities` on write + route the dream-cycle conflict pass through `BlindEntityStore`
  (client-side) under `encryption_mode=blind`.

### Cross-cutting cleanup / E0 follow-up
- Interactive `hermes memory setup`: passphrase prompt (confirm) + tier choice + recovery export +
  no-recovery warning (needs the hermes framework to test).
- Encrypted-DB embed-dim detection relies on the Ollama probe (plaintext `_read_db_vector_dim`
  returns None on ciphertext) — document/handle.
- Deployment: systemd `Environment=` for `RESONANT_LATTICE_DB_ENCRYPTED` / passphrase source;
  key-at-boot (no TPM on the node → once-per-boot SSH unlock, §4.2).

### Open decisions — RESOLVED (2026-06-19, user sign-off)
- **§11 #3 / E7 7a** — DROP FTS in blind mode (vector-only recall); SSE declined. ✅
- **§11 #4** — fact-count leak ACCEPTED for E2 client-side-rank recall (PRE score-handoff added); the
  no-leak path is the pure-CKKS `BlindArgmaxCKKS` (opt-in hardening, not the default recall). ✅
- **E4** non-polynomial ops — RELOCATE to client-side, not polynomial-approx (cos/sin in the lift,
  bundle at encode). ✅
- **Prod params** — `HEStd_128_classic` confirmed (E3 §3b); per-op latency measured on the node. ✅
- **E5 5b** — decay fully blind / promotion-eviction CLIENT-ASSISTED (autonomous declined). ✅
- *Still confirmable (defaults in place):* **§11 #5** `blind_policy` per-cycle scope caps (top-k ≤16,
  ≤8 queries/cycle, ≤64 re-encrypts/cycle) — implemented as defaults in 6c; tune if needed.

### Remaining order
**E7 7b** (entity PSI overlap; 7a = drop-FTS already satisfied by vector-only `BlindRetriever`) → the
DEFERRED PROVIDER WIRING (dream-cycle `decay_all`/`settle`; relational/conflict + 4b through the blind
path; 6e provider PRE-runtime recall + a `memory_audit` action over the re-encryption log) → the full §5
blind store (encrypt content/entities). Discipline (unchanged): node-spike unproven crypto first; commit
per phase; Windows harness green + node-validate the committed code (via the "yes/no sync" question).
