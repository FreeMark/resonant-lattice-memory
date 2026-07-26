# The Interpreter and the Missing Brain

*Design note, 2026-07-19. Part of the "design notes along the way" series.*

---

## Summary

The clearest statement of why this project exists came from the operator, late
in a design session, as an observation about neuroscience rather than about
software:

> "They are language models — let them do language generation. The rest of the
> system is missing. Stop trying to pack all the other pieces into the
> language part. It's well documented in humans that the language side of the
> brain just flat-out makes things up. Its job is to build language from the
> inputs. Input == output."

This note unpacks that observation into the architecture it implies, because
it turns out to be the whole design philosophy of the resonant lattice in one
frame: **an LLM is an interpreter module shipped without the rest of the
brain, and hallucination is not a defect of the module — it is the module,
operating without inputs.**

## 1. The experiment

In the split-brain studies (Sperry's callosotomy patients, analyzed by
Gazzaniga), the two hemispheres can be shown different stimuli. The classic
results:

- The mute right hemisphere is shown the instruction "walk." The patient
  stands and walks. The language-bearing left hemisphere, asked why, answers
  instantly: *"I wanted a Coke."*
- The right hemisphere sees a snow scene; the left sees a chicken claw. The
  left hand (right hemisphere) selects a shovel. Asked why, the language
  hemisphere — which never saw the snow — explains fluently: *"You need a
  shovel to clean out the chicken shed."*

Gazzaniga named the responsible system **the interpreter**: a left-hemisphere
module whose job is to produce a coherent verbal narrative from whatever
information reaches it. When the true cause is available, the narrative is
true. When it isn't, the interpreter narrates anyway — confidently, fluently,
and wrong. It does not lie; it cannot tell the difference. Producing coherent
language from available input is its *entire function*.

## 2. The diagnosis

A large language model is an interpreter module — a very good one — running
with none of the systems that, in a healthy brain, keep the interpreter's
narratives attached to reality:

| In the brain | Function | In a bare LLM |
|---|---|---|
| Hippocampus + consolidation | Episodic memory; what actually happened | absent |
| Anterior cingulate | Conflict monitoring; "these two beliefs disagree" | absent |
| Prefrontal executive | Gating, plan-checking, inhibition | absent |
| Basal ganglia | Trained procedure; how we actually do things here | absent |
| Perceptual reality-testing | Is this narrative consistent with the world | absent |

Ship the interpreter alone and you have engineered the split-brain condition
deliberately. The industry's dominant response — make the interpreter bigger,
fine-tune it harder — improves the *eloquence* of the narration and leaves
the architecture of the problem untouched. A larger interpreter without
inputs is a more persuasive confabulator.

## 3. The design principle

**You cannot stop the interpreter from confabulating. You can only control
what it confabulates from.**

This inverts where the anti-hallucination effort goes. Not into the model —
into the *input side*: what reaches the interpreter's context, with what
provenance, under what verification. The lattice's machinery lines up as the
missing modules, one for one:

| Missing module | Lattice machinery |
|---|---|
| Hippocampus / consolidation | The memory substrate itself: episodic capture → windowed extraction → tiered semantic store, cycle-driven |
| Reality-testing | Grounding contracts: facts require verbatim source quotes; attestation checks the quote against the transcript; no source, no fact |
| Conflict monitoring | Conflict limbo: contradicting facts are held, flagged on recall, arbitrated — never silently merged |
| Procedure | Procedural seeds and distilled tool-lore (with hygiene: an interpreter's misreading of its own tools must not become doctrine — learned the hard way) |
| Narrative identity | A session-summary layer whose inputs are the *banked facts*, so the interpreter's storytelling instinct is fed true material — its confabulation machinery generating true narrative |
| Epistemic scope | Provenance tiers and scoped authority (see the epistemic-constitution note): the store knows what kind of true each input is |

None of this makes the language model honest. It makes the language model's
*inputs* honest, which is the only version of the problem that has a
solution.

## 4. Two corollaries

**Small interpreters suffice.** The human language cortex is a small fraction
of the brain. If the interpreter's job is interpretation — not memory, not
verification, not procedure — then a 2–4B-parameter model with real external
modules should outperform a much larger bare model at long-horizon personal
tasks, because the larger model is only a more fluent narrator of the same
missing inputs. This is the founding bet of the on-device effort: pair a
small interpreter with the full external cognitive loop and let each part do
its job.

**Let the language module do language — everywhere, including output.** The
same principle applies at the output boundary. Rigid serialization formats
(JSON) force the interpreter to run a syntax ledger alongside content
generation — a coherence tax, paid in the model's scarcest resource, with
binary failure modes. The companion decision (same day): natural
line-oriented record formats, grammar-constrained decoding where the runtime
bears the syntax burden, or prose-then-lift where deterministic code does the
formatting. The interpreter interprets; structure is someone else's job.

## 5. The one-sentence version

The industry is trying to teach the narrator to stop narrating. This project
builds the narrator the brain it was amputated from — and then feeds it only
things from its own experience.

*(An earlier draft said "things that are true." The operator corrected it,
and the correction is the deeper claim: the system never certifies truth —
it certifies provenance. Attestation checks that a quote exists in the real
transcript, not that the world agrees with it. The provenance tiers are
experiential modes — witnessed, read, was-told — and remembering "the
operator said X" stays honest even when X turns out to be wrong. Experience
is the category a memory can actually keep. Truth was always too big a
promise.)*
