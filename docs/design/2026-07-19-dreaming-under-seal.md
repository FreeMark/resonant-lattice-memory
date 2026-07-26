# Dreaming Under Seal

*Design note, 2026-07-19. Part of the "design notes along the way" series.
Status: blueprint, refined from an operator first-draft the same evening.*

---

## The problem

The encrypted-store roadmap has one standing wound: consolidation. Blind
recall is solved (CKKS cosine top-k over sealed vectors, GPU-accelerated,
shipped). But the dream cycle — decay, merge, gist, abstraction — has always
seemed to require decrypting the store, because dreaming was assumed to be a
text operation: an LLM reads facts, writes conclusions. The privacy seam was
therefore "decrypt everything, dream, re-seal" — a trusted-zone excursion for
the entire memory, every cycle.

## The observation that splits it

Dreaming is not one operation. Inventory the cycle:

| Dream operation | What it actually computes | HE-compatible? |
|---|---|---|
| Decay / dwell / promotion | scalar multiplies on resonance | YES — CKKS native |
| Near-duplicate merge | cosine threshold + weighted centroid | YES — shipped machinery |
| Clustering (abstraction candidates) | similarity geometry | YES |
| Conflict detection | similarity + polarity flags | mostly |
| Gist / abstraction / narrative TEXT | generative language | NO — the hard kernel |

Most of the dream is geometry, and the sealed store already speaks geometry.
Only the *verbalization* of dream products needs language.

## The blueprint: dream in vectors, wake with conclusions

1. **Geometric dream under seal.** Decay, dwell, merges, clusters, and
   gist-candidates run entirely on encrypted vectors in the frozen embedder
   space. A merged concept is a weighted centroid computed blind. Nothing
   decrypts. (Encrypted-resonance-as-source-of-truth is the bookkeeping half
   of this, already on the roadmap.)
2. **Decrypt derivatives only.** At wake, the cycle's *products* — a handful
   of new vectors and bookkeeping deltas — are decrypted. The source facts
   are not. The seam shrinks from "the whole store crosses into the clear
   every cycle" to "only what the dream concluded crosses, once."
3. **Verbalization, two paths:**
   - **Ecosystem inverter (preferred):** a vec2text-style inversion model
     trained on the frozen embedder. Because the embedder is frozen by
     invariant, this is ONE trained asset for the entire ecosystem —
     train once, ship alongside the embedder, works on every install.
   - **Per-model adapter W:** a linear projection from frozen-embedder
     space into the chosen interpreter's input-embedding space, so dream
     products arrive as soft tokens the model verbalizes directly. A
     linear map is CKKS-native, so the projection itself can run under
     seal. Model-lock — the operator's instinct that "you must pick a
     model to use the HE layer" — collapses to W alone: the lattice's
     vector space stays portable; switching interpreters means retraining
     a cheap linear probe, not re-embedding a life.
4. **The far rung, named honestly:** true encrypted forward passes
   (embedding layer and linear ops are HE-cheap; softmax and layernorm are
   the walls — polynomial approximation + bootstrapping territory).
   Research-horizon, not roadmap. The blueprint above needs none of it.

## The conservative flag

Gist-by-centroid must PROVE it preserves exceptions before verbal
abstraction leaves the trusted zone. The abstraction law is
contextualization-not-erasure; a centroid is by construction a smoothing
operator. Until vector-space dream products demonstrably keep the
"default + exception" structure (measurable: the same expect-required
battery the reasoner bench uses), text-level abstraction stays at the seam,
inside the trusted zone. The seam shrinks; it does not vanish by assertion.

## Prior art, honestly named

CKKS linear algebra (the shipped blind-recall backend); vec2text-style
embedding inversion (Morris et al.); soft-prompt / prefix conditioning;
linear probes between representation spaces; encrypted-inference research
for the far rung (minutes-per-pass BERT-tiny class). The assembly — a
sealed geometric dream whose products alone are decrypted and verbalized
through a frozen-space inverter — appears to be the contribution.

## One-sentence version

Seal the memories, let the dream do its arithmetic blind, and when it wakes,
let it speak only its conclusions — the diary stays locked even from the
dreamer.
