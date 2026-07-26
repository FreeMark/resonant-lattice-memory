# Dreaming Under Seal — Build Plan

**Status:** blueprint → engineering plan, 2026-07-19 (same-night refinement of an
operator first-draft). UNCOMMITTED until the operator's call, like the design-note
series. Companion philosophy note: `docs/design/2026-07-19-dreaming-under-seal.md`.
Parent roadmap: `resonant_lattice/ENCRYPTION_ROADMAP.md` (the E-phases / §5-x memo) —
this plan is the missing consolidation lane between encrypted-resonance-as-SOT and
THE SEAL. Scope: FULL RLM only; pocket-lattice stays SQLCipher-at-rest by standing
decision.

---

## 0. The claim, precisely

Consolidation ("dreaming") does not require decrypting the store. Most of the dream
is geometry — and the sealed store already speaks geometry. The build splits the
cycle into a **sealed geometric dream** (runs blind on encrypted vectors), a
**wake-side verbalizer** (turns decrypted *dream products* — never source facts —
into language), and a **residual trusted-zone kernel** (verbal abstraction, conflict
arbitration) that shrinks as gates are passed but is never assumed away.

The privacy delta this buys: today's seam is "decrypt everything to dream, every
cycle." The target seam is "scalar scores cross during the protocol; product
vectors cross once, at wake; source facts never cross again after sealing."

**Non-goals (explicit):** FHE transformer inference (the far rung — embedding/linear
layers are CKKS-cheap, softmax/layernorm are research walls; nothing below depends
on it). Replacing the frozen embedder. Any pocket-lattice coupling.

---

## 1. What exists today (inventory with receipts)

| Asset | State | Where |
|---|---|---|
| CKKS blind cosine top-k, GPU-accelerated (~50×, bit-exact vs CPU, ~7s @ 4952 facts) | SHIPPED v1.4.2, public MIT | `retrieval.py` GPUBlindBackend + `rlm_gpu_recalld` daemon; github.com/FreeMark/openfhe-gpu-backend |
| Blind store substrate: `semantic_he*` tables, `reencrypt_audit`, `facts_missing_blind` worklist | SHIPPED (default-off) | `store_blind.py`, `store_schema.py` |
| Client-visitor foundation (§5-2) | BUILT/TESTED/PUSHED, inert | canonical main `127193a` lineage |
| Frozen embedder (embeddinggemma, 768-dim; Matryoshka 256 truncation) | ECOSYSTEM INVARIANT | everywhere; the space all HE work is dimensioned on |
| Dream-cycle operations (decay, dwell, promotion, near-dup merge, conflict bleed, prune, long-tier cap, gist) | SHIPPED, plaintext | `store_dream.py`, `store_abstraction.py` |
| Substrate-asserting encryption test suite (at-rest opacity, 0.95 merge, conservative conflict) | 8 PASS | `grok/_harness/testing2/` |
| Exception-preservation battery (expect-required token groups; fabrication hard-token gate) | SHIPPED in the pocket bench | `pocket-lattice/bench/` (reusable pattern) |
| Three-key / PRE envelope concept (owner-inspectable interior) | DESIGNED (§5-5, not built) | encryption memo |

Key crypto fact the whole plan leans on: **CKKS evaluation keys allow a host to
COMPUTE without the ability to DECRYPT.** The dream host holds eval keys only; the
keyholder (owner/client) holds the secret key. This is standard CKKS key topology —
no new cryptography is invented anywhere in this plan.

---

## 2. Architecture

```
           SEALED STORE (frozen-embedder vectors, CKKS)
                │
   ┌────────────▼─────────────┐
   │ C1 Sealed Geometric      │   host-side, eval keys only
   │    Dream Engine (SGDE)   │   decay · dwell · frontier similarity ·
   │                          │   centroid construction · cluster stats
   └───────┬───────────┬──────┘
           │ encrypted │ encrypted scalar
           │ products  │ scores (masked)
           ▼           ▼
   ┌───────────┐   ┌──────────────────┐
   │ C2 Dream- │   │ keyholder round: │  client-side, secret key
   │  Product  │   │ decrypt scores,  │  threshold decisions ONLY
   │  Ledger   │   │ return decisions │  (merge yes/no, promote yes/no)
   └─────┬─────┘   └──────────────────┘
         │ wake: decrypt PRODUCTS only
         ▼
   ┌──────────────────────────┐
   │ C3 Wake-side Verbalizer  │  path A: ecosystem inverter (vec2text
   │    + DERIVED provenance  │  on frozen embedder — ONE trained asset)
   └─────┬────────────────────┘  path B: linear adapter W → soft tokens
         │
         ▼
   trusted-zone residue: verbal abstraction, conflict ARBITRATION,
   narrative — unchanged today, shrinks only by passing gates
```

**C1 — Sealed Geometric Dream Engine.** All ops expressible as CKKS linear algebra:
- decay/dwell: batched scalar multiplies over the encrypted resonance vector
  (this IS encrypted-resonance-as-SOT, already on the roadmap as §5-3).
- similarity: reuse the shipped blind top-k. Dream merging does NOT need the N×N
  matrix: merges overwhelmingly involve the recent frontier, so candidate
  generation = frontier facts (this cycle's additions/reinforcements) × store —
  the exact 1×N shape the recall backend already serves. **Frontier-limited
  merging is a design rule, not an optimization** (it also bounds leakage surface).
- centroids: encrypted weighted averages (plain CKKS mults+adds).
- cluster stats for abstraction candidates: same machinery.

**C2 — Dream-Product Ledger.** New encrypted vectors (merge centroids,
gist-candidates) + plaintext lineage metadata (which fact IDs merged, cycle number).
IDs and graph shape are metadata — see threat-model deltas (§5, G3).

**Keyholder rounds.** CKKS has no native comparison. Thresholding (merge τ,
promotion cutoffs) is resolved by an **interactive round**: host sends encrypted
(and mask-blinded) scores; keyholder decrypts scalars, returns yes/no decisions.
Scalars cross the seam; content never does. Deployments without a present keyholder
either batch decisions until the keyholder next appears ("**the dream finalizes
when the phone visits**") or pay for polynomial sign-approximation (known technique,
deep circuits + bootstrapping — priced in §6, not required for v1).

**C3 — Wake-side Verbalizer.** Two independently useful paths:
- **Path A (preferred, ecosystem asset): frozen-space inverter.** A vec2text-style
  inversion model trained once against embeddinggemma; because the embedder is
  frozen by invariant, one inverter serves every install forever. Ships beside the
  embedder like a codec.
- **Path B (per-model adapter W):** linear projection frozen-space → interpreter
  input-embedding space; dream products arrive as soft tokens the interpreter
  verbalizes. W is a cheap linear probe (re-train per interpreter; model-lock
  collapses to W — the operator's "you must pick a model" instinct, contained).
  Bonus: a linear map is CKKS-native, so the projection can itself run under seal
  and decrypt directly into model-space.

**Provenance: the DERIVED tier (constitution change).** A verbalized dream product
cannot carry a source quote — its attestation is **lineage, not quotation**:
`derived(cycle=N, from=[ids], op=merge|gist)`. Recall must LABEL derived facts as
derived. The epistemic constitution gains one tier; the fabrication gate still
applies to verbalizations (no hard tokens absent from the lineage's source facts —
checkable at wake inside the trusted zone, where sources are inspectable by the
keyholder).

---

## 3. The ledger: solved / partial / open

**SOLVED (shipped, ours):**
- Blind cosine at scale, GPU, bit-exact (the entire similarity substrate of C1).
- Encrypted vector storage, rotation, re-encryption audit.
- Frozen-space dimensioning incl. Matryoshka truncation.
- At-rest opacity testing methodology (subprocess + `encrypted_binding_active()`).

**SOLVED IN LITERATURE (needs our build, no research risk):**
- Weighted centroids / linear maps under CKKS (trivial circuits).
- Embedding inversion (vec2text lineage) — training project, one asset.
- Linear adapters between representation spaces; soft-prefix conditioning.
- Interactive threshold protocols (standard blinded-decrypt rounds).

**PARTIAL (design decided, engineering open):**
- Comparison under seal: interactive round chosen for v1; sign-polynomial fallback
  priced but deferred.
- Conflict handling: DETECTION blind (similarity + stored polarity flags);
  ARBITRATION stays trusted-zone (it is a semantic judgment).
- Dwell/promotion bookkeeping: scalar math blind; cutoffs via keyholder rounds.

**OPEN GAPS (each with its gate):**
- **G1 — Exception preservation of vector gists.** A centroid is a smoothing
  operator; the abstraction law is contextualization-not-erasure. GATE: the
  expect-required battery (pocket bench pattern) run against centroid-gists vs
  text-gists. Until centroids demonstrably keep default+exception structure,
  verbal abstraction stays at the seam. Expected outcome: centroids handle
  near-dup merge (safe) but NOT multi-condition abstraction (stays trusted).
- **G2 — Verbalizer fidelity.** Inverters can hallucinate. GATE: zero fabricated
  hard tokens vs lineage sources (the existing fabrication gate, applied at wake);
  fidelity threshold on round-trip (text → vector → text) before any derived
  verbalization banks.
- **G3 — Metadata leakage.** Lineage graphs, timing, counts, frontier sizes leak
  structure even when content is sealed. DELIVERABLE: a threat-model delta doc
  (honest-but-curious host reading: what the graph shape reveals; mitigations:
  padding, batched cycles, dummy products). Never claim "the host learns nothing";
  claim precisely what it learns.
- **G4 — Key topology in production.** Eval-keys-only host is standard; the
  interplay with the three-key PRE envelope (§5-5) needs one worked design:
  who holds eval keys, who attends keyholder rounds, what the owner-inspectability
  guarantee looks like mid-dream. (Alignment note: the visible-boundary principle
  is PRESERVED — the keyholder can always open everything; the agent knows its
  products are derived-under-seal.)
- **G5 — Cost model.** Centroid/decay circuits are shallow (no bootstrapping);
  frontier×store similarity reuses the measured recall path (~7s @ ~5K facts on
  the node GPU → a nightly frontier of ~50 facts ≈ same order). Sign-approx path
  would be 10-100× that. MEASURE in D1/D2; no dial-turning on estimates.
- **G6 — What never moves (v1 honesty):** narrative, text abstraction, conflict
  arbitration, anything requiring reading content — trusted zone, by design,
  until G1/G2 gates pass for their specific op.

---

## 4. Build phases (gated; numbers or it didn't happen)

- **D0 — PROTOCOL PAPER.** Write the interactive-round protocol + threat-model
  delta (G3) + key topology (G4) as one reviewed doc. No code. EXIT: operator
  review; deployment-class matrix agreed (§5).
- **D1 — SEALED BOOKKEEPING (= §5-3 encrypted-resonance-as-SOT, subsumed).**
  Resonance/dwell as CKKS vectors; decay = batched scalar mult; promotion via
  keyholder round. EXIT: decay curves bit-match plaintext reference on the eval
  corpus; at-rest opacity suite extended and green; cycle cost measured.
- **D2 — BLIND MERGE PIPELINE.** Frontier-limited candidates via the shipped
  blind top-k; keyholder threshold round; encrypted centroid write + lineage
  ledger (C2). EXIT: merge decisions ≡ plaintext dream on the corpus (agreement
  ≥ 0.98, zero merges below τ), frontier cost within 2× of a recall pass.
- **D3 — THE G1 GATE (exception preservation).** Centroid-gist vs text-gist on
  the expect-required battery. EXIT: a DECISION with receipts — which ops earn
  blind status (expected: merge yes, multi-condition abstraction no).
- **D4 — VERBALIZER TRACK (parallel to D1-D3).**
  D4a: adapter W for granite-8b and E4B (linear probe; soft-token verbalization
  demo). D4b: the ecosystem inverter (vec2text on embeddinggemma; the one-asset
  bet). EXIT: round-trip fidelity threshold + zero fabricated hard tokens (G2
  gate) on a held-out set; pick A/B/both per results.
- **D5 — INTEGRATION.** `sealed_dream` config (default-off, inert-shipped like
  every encryption phase); DERIVED provenance tier lands in the constitution +
  recall labeling; business-test suite extended (blind-merge + derived-recall
  cases); node deployment behind the standing test-gated rlm-push.

Standing rules apply throughout: phase-gated with operator go-ahead per phase,
substrate verification at the SQLite/HE layer, default-off shipping, core-first
(this is core RLM work; nothing grok- or profile-specific).

---

## 5. Deployment classes (who dreams, where the keyholder is)

| Class | Example | Keyholder rounds | Notes |
|---|---|---|---|
| Self-hosted, single box | today's node deployments | trivial (keys local, seam = process/at-rest boundary) | v1 target; interactive rounds are function calls |
| Client + host pair | hosted lattice, owner's device holds keys | batched at check-in — "the dream finalizes when the phone visits" | the honest async model; products queue sealed |
| Always-on hosted, absent keyholder | web-served growing lattices | requires sign-approx (G5 cost) or delegated threshold keys (G4) | v2+; do not promise before pricing |

---

## 6. Cost sketch (to be replaced by D1/D2 measurements)

Shallow circuits only in v1: decay (1 mult/fact), centroids (k mults + adds),
frontier similarity (= recall passes, measured at ~7s per ~5K-fact sweep on the
node's GPU backend; a nightly frontier of ~50 changed facts is the same order).
No bootstrapping anywhere in the v1 path — bootstrapping enters only with
sign-approximation (deferred) or deep inverter-under-seal fantasies (not planned).
Rule inherited from the house: measure before believing; the estimates above are
for scoping, not for commitments.

---

## 7. One-paragraph summary for future-us

The dream was never one thing. Its arithmetic — decay, merge, clustering — runs
blind today on machinery we already shipped for recall; its judgments — thresholds
— cross the seam as naked scalars in an interactive round; its products — a
handful of new vectors — decrypt once at wake and are spoken by a verbalizer that
never sees a source memory (one frozen-space inverter for the whole ecosystem, or
a per-model linear adapter that contains the model-lock to a single trainable
matrix). What stays in the trusted zone stays because a gate said so, with
receipts: centroids must prove they keep exceptions before verbal abstraction
moves, and every verbalization passes the fabrication gate against its lineage.
The diary stays locked even from the dreamer; the dreamer wakes fluent only in
its conclusions.
