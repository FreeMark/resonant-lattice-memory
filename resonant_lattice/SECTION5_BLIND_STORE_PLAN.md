# §5 Full Blind Store - Plan

> Status: PLAN with all four §3 design decisions LOCKED (user sign-off
> 2026-07-10). Phases not yet started; §5-0 spikes gate everything. Companion to
> ENCRYPTION_ROADMAP.md (whose §5 table defines the encryption surface and whose
> §14 lists this as the horizon). Everything here rides the BlindTier seam and
> the locked decision log: minimal leakage, client-assisted over
> store-autonomous-with-leakage, cycles-not-seconds absorbs the cost.

## 1. What §5 actually is

Today `encryption_mode=blind` is a MIRROR: the encrypted tables (`semantic_he`,
`semantic_he_hrr`, `semantic_he_entities`, `semantic_he_meta`) sit beside a fully
plaintext store, recall runs blind to prove the compute path, and
`_blind_reconcile()` keeps the mirror complete by reading plaintext back.

§5 flips the source of truth. After §5, a blind store holds:

| Data | Post-§5 state |
|---|---|
| fact content, source_quote, source_ref, category text | AEAD ciphertext (`semantic_he_content`) |
| exact-dup identity | keyed HMAC of normalized content (UNIQUE) - store can match, never dictionary-attack |
| embeddings / content-HRR | CKKS ciphertext (already done) |
| entity sets | AEAD (already done) |
| relation triples | AEAD text + the existing per-triple HRR path; graph walks move client-side |
| resonance | CKKS ciphertext in `semantic_he_meta`, decay-from-origin - SOURCE OF TRUTH (6b lands inside §5) |
| episodes, tool episodes, session summaries, agent identity | AEAD rows |
| FTS index | not built (locked E7 decision) |
| tier / decay-class / limbo / quarantine | encrypted control vectors: one-hot decay class + include-flag (3b, DECIDED) |
| conflict_group_id | AEAD metadata (store never uses it post-§5-2) |
| cycle clock, set_cycle, row counts, insertion/update timing | PUBLIC (irreducible; documented leakage, §6) |

The store keeps computing (blind recall, blind HRR scoring, blind decay, blind
resonance bumps); everything that inherently needs plaintext (LLM extraction,
embedding, entity extraction, HRR encode, attestation, clustering, adjudication,
gisting) already happens in the agent process and simply stops writing plaintext.

## 2. The one architectural shift: the dream cycle becomes a client-side visitor

Today the dream cycle is store-side code reading plaintext rows. Under §5 the
store cannot read anything, so every cognition pass that needs content becomes a
CLIENT visitor over decrypted working sets:

- clustering/abstraction: client decrypts candidate HRR lifts + entity sets,
  clusters, calls the reason model, writes new abstract facts encrypted;
- conflict detection: similarity band from blind HRR scores (store-side) or
  client-side after decrypt; entity-overlap gate via `BlindEntityStore.
  find_conflicts` (built); adjudication is an LLM call (client by definition);
- procedural distillation / gist / narrative: already LLM-bound → client;
- promotion/eviction: `BlindMaintainer.settle()` client-assisted (locked E5 5b).

What stays store-side and blind: vector recall, HRR scoring, decay, resonance
bumps, dedup-candidate scoring, and (optional hardening) argmax. This is the
same split the E-phases proved piece by piece - §5 just re-plumbs the dream
cycle to route through it.

## 3. Design decisions - ALL FOUR LOCKED (user sign-off 2026-07-10)

> (3a) agent-key AEAD v1, PRE envelope only if the §5-0a sizing spike is sane;
> (3b) encrypted-control-vector design is v1 pending spike §5-0(d), public
> metadata is the fallback; (3c) AEAD triples + client-side graph walks, the
> HMAC-token alternative DECLINED ("keep it unidentifiable" - co-occurrence
> leakage, same class as the declined entity PSI and SSE); (3d) dedup strategy
> decided after the backfill experiment's real numbers. Don't re-litigate;
> rationale below.

**3a. Content key model - the only genuinely new crypto question.**
- **v1 (recommended): agent-key AEAD.** A master-derived content key (HKDF ctx
  `"content-v1"`), agent-wrapped in the keystore like the entity key. The STORE
  node never has it; the agent decrypts content of recalled facts. Honest seam:
  an agent that can read the DB file could decrypt all content - for content the
  "use but can't read" bound is policy (scope caps + audit), exactly like
  tension 7.2. Ships with zero new crypto.
- **v2 (hardening, phase 5-5): envelope + PRE.** Per-fact random data key,
  CKKS-encrypted under master; on recall the store PRE-flips the key ct to the
  agent use-key through the existing token gate (`BlindReEncryptGate`), so every
  content unlock is a single-use, audited event and the math bound returns.
  BLOCKER to check first: ct size for a key-wrap context (E2's 787KB/ct would be
  ~2GB overhead at 2500 facts - needs a small-ring spike, §5-0a).

**3b. Public metadata set - UPGRADED (user asked "can any of this go behind
HE?"; answer: mostly yes).** Encrypted-control-vector design, spike §5-0(d):
- **Decay-class/importance BEHIND HE:** per-fact encrypted ONE-HOT class vector;
  decay = homomorphic inner product with the PUBLIC per-class scalar vector
  [f1^t, f2^t, 1.0] then multiply resonance ct. Depth 2, constant in t; the
  store applies the right rate without learning which.
- **Tier BEHIND HE:** tier reduces to (a) decay exemption = the factor-1.0 row
  of the same one-hot, and (b) recall exclusion (superseded/dormant/quarantine)
  = an encrypted 0/1 include-flag multiplied into the score ct (+1 depth on
  recall). Conflict-limbo decay skip = a class; quarantine = the flag.
- **conflict_group_id:** moves to AEAD metadata (store never uses it once the
  conflict passes are client visitors, §5-2).
- **Genuinely public (irreducible):** the cycle clock (shared logical time) and
  row existence/insertion timing/update patterns - HE hides values, not access
  patterns (ORAM is out of scope). Encrypting set_cycle is possible (bake
  factor^(-set_cycle) into the ct client-side, rebase at settle) but pointless:
  insertion timing already reveals it.
Cost: one small extra ct per fact, maint keyset depth 2 (~2.6MB vs 0.8MB),
+1 recall depth. RECOMMEND: v1 ships the encrypted-control-vector design if
spike §5-0(d) is clean; public tier/decay-class metadata is the fallback.
Leakage profile in §6 updated accordingly.

**3c. Relation graph.** Recommend AEAD triple text + client-side exact graph
walks (decrypt triples of facts already in play), keeping the blind fuzzy-HRR
triple scoring as is. The store-side alternative (deterministic HMAC tokens for
subject/object equality) leaks co-occurrence - the same leakage class as the
declined entity PSI and SSE, so consistency says decline it too. Cost: `infer`
chains become client work; bounded hops keep it cheap.

**3d. Dedup at scale.** Semantic dedup (0.78) and near-identity reinforce (0.95)
become blind-recall queries at write time. Linear blind scan is ~26ms/fact →
~65s per new fact at 2,500 facts, linear in corpus size. Consolidation is off
the hot path (cycles-not-seconds), but 20 facts/session at 10k facts would be
~87 min/epoch. Levers, in preference order: (i) batch the epoch's new facts into
one scan pass (score all K new probes per stored ct in one sweep - K× fewer ct
loads); (ii) a trusted-CLIENT embedding cache (the agent box is trusted; an
at_rest-encrypted local ANN cache for dedup only, authoritative store stays
blind); (iii) accept the grind. DECIDE AFTER the queued backfill experiment
produces real numbers (§5-0c is free).

**3e. UNIQUE and reinforce identity.** `content UNIQUE` → `content_hmac UNIQUE`
(HMAC-SHA256 under a derived key, normalized content). Exact-dup insert resolves
store-side with zero leakage beyond "a duplicate exists," which the plaintext
store leaks today anyway. Reinforce (resonance bump on the existing fact) is a
homomorphic ADD on `semantic_he_meta` - spike §5-0b.

## 4. Phase plan (house rhythm: gated, test-backed, substrate-validated, default-OFF)

**§5-0 - Spikes (node, SSH heredoc, no sync needed). ALL DONE 2026-07-12.**
- (a) DONE. Key-wrap sizing: BGV 128-bit depth-0 ring-4096, exact PRE roundtrip,
  129 KB/fact (6x under the feared 787 KB); ~6-7% vault growth. 3a-v2 size-viable
  -> kept for §5-5; 3a-v1 AEAD ships first.
- (b) DONE. Homomorphic resonance ADD = depth-0 plaintext add of the public scalar
  b*factor^(origin-c); exact ~1e-13 within a settle window, rebase-at-settle keeps
  it unbounded. No new crypto, no depth cost; read stays depth-1.
- (c) DONE (earlier, backfill experiment): ~356s/query CPU -> A1 streaming + GPU
  (~7s). Dedup-at-scale tractable via batched-epoch + GPU.
- (d) DONE. Encrypted-control-vector (3b): class-hidden decay (one-hot x public
  rate vector, depth 2) exact ~1e-13 incl. the f=1.0 pinned class; include-flag
  (x score, +1 recall depth) masks exactly. Tier/decay-class/limbo/quarantine move
  behind HE. Cost: maint keyset depth 2 (~2.6MB), +1 ct/fact, +1 recall depth.

**§5-1 - Sealed-content mirror (additive, non-destructive).**
Keystore v3 (+content key, +episode key, +triple key); new tables
`semantic_he_content`, `semantic_he_episodes`, `semantic_he_triples`,
`semantic_he_summaries` (+`content_hmac` column, backfilled); `_blind_reconcile`
extended to mirror ALL of it. Store stays plaintext-authoritative; suite +
substrate checks prove the mirror is complete and opaque. Migration stays
idempotent.

> STATUS 2026-07-12 - §5-1 COMPLETE (all four surfaces). §5-1a CONTENT (the anchor):
> `semantic_he_content` (AEAD `{content, category, source_quote, source_ref}`,
> random-nonce opaque), `semantic_facts.content_hmac` (keyed HMAC dedup identity, 3e)
> + partial index, `retrieval.BlindContentStore`, content-mirror + hmac-backfill
> reconcile. §5-1b TEXT surfaces: `semantic_he_episodes` ({role,content} <- episodes),
> `semantic_he_triples` ({subject,relation,object} <- fact_relations),
> `semantic_he_summaries` (summary text <- session_summaries), each keyed by its SOURCE
> row (CASCADE-FK) with its own idempotent LEFT-JOIN worklist + payload reader in
> BlindMixin, mirrored via `retrieval.BlindSealedStore`. Shared crypto:
> `encrypt_sealed`/`decrypt_sealed` (domain-parametrized AEAD), `derive_sealed_keys`
> (ALL keys in one Argon2id pass), `content_hmac`. DESIGN NOTE: "keystore v3" is
> realized as HKDF SIBLING keys derived on demand (like the entity key), NOT a
> keystore-sidecar version bump - the sidecar format is unchanged, so existing keystores
> keep working and `keystore_is_secret_free` still holds. 4 new tests, suite 130 -> 134
> green; Windows-substrate-validated (pure AEAD/HMAC, no openfhe - same bar as the E7
> entity AEAD); migration idempotency confirmed on re-open. NOT deployed.

**§5-2 - Client-visitor dream cycle (the big code motion).**
Behind the existing mode flag, dream-cycle passes route through a BlindTier
visitor API (decrypt working set → run existing pass logic → write encrypted).
Plaintext-mode code paths untouched. Acceptance: blind-mode dream cycle
reproduces plaintext-mode outcomes on a fixture DB (same promotions, same
conflict groups, same abstractions given the same LLM stubs).

> STATUS 2026-07-12 - §5-2 FOUNDATION LANDED: the client-visitor read surface +
> its parity acceptance. `retrieval.BlindVisitor` (via `BlindTier.visitor()`)
> assembles each fact's dream-cycle WORKING SET - `fact_view` = {content, category,
> quote, ref, entities}, plus `triples(fact_id)` (relation ids read structurally,
> text decrypted from `semantic_he_triples`), `summary(id)`, `episode(id)` - entirely
> from the §5-1 sealed ciphertext. `store_blind.relation_ids_for_fact` gives the
> structural fact->relation map that survives the seal. Parity test proves the
> blind-served working set == the plaintext store's, AND that it is truly
> blind-sourced (tampering the plaintext `content` column does not change the
> visitor's output - it reads ciphertext). This is the load-bearing mechanism +
> acceptance for the READ half: identical working set + fixed LLM stub => identical
> pass outcomes by construction. 1 new test, suite 134 -> 135 green.
> REMAINING §5-2 (couples with §5-3/§5-4, next sessions): re-route each real
> consolidation pass (abstraction/gist/conflict-adjudication/narrative) to CALL the
> visitor instead of reading plaintext directly, behind the mode flag, so the dream
> cycle runs with NO plaintext; then the end-to-end fixture-DB parity (same
> promotions/conflict-groups/abstractions). NOT deployed.

**§5-3 - Encrypted resonance becomes source of truth (absorbs roadmap 6b).**
`set_cycle` column + decay-from-origin `BlindMaintainer` (depth-1, locked);
recall bump = blind ADD (from §5-0b); public decay-class drives importance
weighting; plaintext resonance column ignored in blind mode (kept until seal).
Promotion/eviction via client `settle()` each dream cycle.

**§5-4 - THE SEAL (destructive, explicitly user-gated).**
A `seal` harness command: verify per-fact completeness (content+embedding+HRR+
entities+resonance cts all present) → verify blind recall parity on a probe set
→ force a god-mode export (decrypt-to-plaintext backup, user-held) → then NULL
all plaintext content/quote/entity/triple/episode text, drop `semantic_fts`,
VACUUM. Post-seal substrate audit: dump every TEXT column in the file, assert
no natural-language content survives; plain `sqlite3` sees structure but reads
nothing meaningful. Reversible ONLY via the export taken at seal time.

**§5-5 - Optional hardening: PRE content envelope (3a-v2).**
Only if §5-0a sizing is sane. Token-gated, audited content unlocks; completes
the math-bound three-key story for content.

**§5-6 - Horizon (explicitly beyond §5): physical split.**
Store daemon holding eval keys only, agent client over a local socket - the
"external/blind hardware" deployment. §5 makes every row and every store-side
computation ready for it; the daemon is packaging, not new crypto.

## 5. Migration story for a trained DB (e.g. webdev/nemo lineage)

1. (available today) at_rest-seal a copy via `sqlcipher_export` - the
   `encrypt_existing_db` helper ships with the queued backfill experiment.
2. §5-1 mirror backfill (extends the proven `_blind_reconcile` path).
3. Verify + operate hybrid as long as desired (recall already blind).
4. `seal` when confident (god-mode export in hand).
So "train in the clear, seal afterwards" becomes a first-class, staged workflow
- the queued experiment validates stages 2-3 at 2,500-fact scale before any §5
code exists.

## 6. Documented leakage profile after §5 (honest-seam summary)

Store/node operator learns: row count and growth rate; write/recall timing and
access patterns (incl. which rows the client updates at settle); ciphertext
sizes; the cycle clock; WHICH fact ids a query's re-encryptions touched (audit
is the point). With the 3b encrypted-control-vector design, tier/decay-class
distribution and conflict-group structure are NO LONGER visible (they were in
the public-metadata fallback, which would add them back - documented tradeoff). Store cannot learn: content, quotes, sources,
categories-as-text, entity names or co-occurrence, triple text, embedding or
HRR geometry, resonance values, query text. Compromised-agent blast radius:
bounded by `blind_policy` scope caps + `reencrypt_audit` (policy, not math) -
unchanged from the locked E6 posture; §5-5 tightens content to math-bound.

## 7. Test/validation additions

Per-phase substrate queries as always, plus: a `blind_parity` fixture harness
(same fixture, plaintext vs blind mode, assert identical recall/promotions/
conflicts); an opacity audit reusable by the seal (entropy/dictionary scan over
all TEXT/BLOB columns); keystore v3 secret-freeness test; migration idempotency
re-runs. Node-validated per the standing seam (helpers at substrate; provider
glue harness-validated).

## 8. Suggested order of operations from today

1. Run the queued webdev blind-backfill experiment (feeds §5-0c numbers).
2. §5-0a/0b spikes over SSH (one evening).
3. Decide 3a/3b/3c/3d with the numbers in hand.
4. §5-1 onward, one gated phase per session, house rhythm.
