# The Epistemic Constitution: Teaching a Memory What Kind of True Its Facts Are

*Design note, 2026-07-19. Part of the "design notes along the way" series.*

---

## Summary

A persistent agent memory eventually holds knowledge from several sources of
very different character: the operator's own hard-won expertise, curated
imports from research corpora, and raw web-derived material. This note records
the day the system's naive authority rule — "the operator's testimony always
wins" — was found to contain a subtle poison class, and was replaced by a small
constitution: **provenance tiers, scoped authority, a principle/procedure
split, and conditions that travel with facts.** The refinement was proposed,
implemented, and made self-enforcing within the same day, using machinery the
lattice already had.

The general claim, for anyone building agent memory: *storing facts is the
easy part. Encoding what kind of true each fact is — for whom, under what
conditions, at what level of execution — is what separates a knowledge base
from a liability.*

---

## 1. The setup: three kinds of knowledge in one store

By the time this note was written, the lattice routinely held facts from three
provenance classes:

| Tier | Source | Character |
|---|---|---|
| **Operator primary** | Direct testimony from the system's operator, an expert practitioner | High-value, experience-dense, low-volume |
| **Curated import** | Selective transfer from research lattices built by overnight web-research farms, tagged `import:<lattice>:<id>` | Broad, sourced, textbook-grade |
| **Web extract** | Raw research output before curation | Wide, uneven, consensus-flavored |

The first authority rule was the obvious one: on conflict, operator primary
wins. It is *almost* right, and its failure mode is invisible until you look
for it.

## 2. The trigger: the operator's own concern

The refinement did not come from a failure. It came from the operator raising
a lifetime observation about their own knowledge, paraphrased:

> "What works for me often doesn't work for others, because what works for me
> requires explicit actions to work extremely well. My concern is that
> something I know to be accurate — in a very explicit way — could be
> considered incorrect data amongst the masses."

This is the honest self-assessment of a deep practitioner: expert optima are
frequently **N=1 optima**. They are reached by compounding many explicit,
correctly-executed steps ("a hundred things later..."), and they genuinely
outperform consensus practice — *at full execution of the entire stack*.
Handed to a median user at median execution, the same technique fails, and the
"fact" reads as wrong.

The inverse is equally true: consensus advice is not wrong. It answers a
different question — *what works robustly for the median practitioner?* —
and for most audiences it is the correct recommendation.

Neither source is "the correct data." They are answers to **different
questions about different populations**, and a memory system that stores both
without encoding that distinction will eventually serve one as if it were the
other. We named this failure mode **context-stripping**: a conditional truth
with its conditions removed becomes false in most hands while remaining true
in its original context.

Context-stripping joins the system's existing poison taxonomy alongside
fabrication (facts with no source), staleness (facts that outlived their
world), and tool-scars (an agent's misread of its own tooling banked as lore —
see the operational history for that incident). It is the subtlest of the
four, because every individual fact involved is *true*.

## 3. A case study (anonymized)

The concrete trigger domain was multi-color printing of flexible filament
(TPU) — one of the harder disciplines in FDM manufacturing, and the operator's
professional differentiator. Two examples:

**Consensus knowledge (median-execution truth):** flexible filament is managed
with conservative retraction settings, reduced speeds, and a direct-drive
extruder. This is correct, robust advice for nearly everyone.

**Expert-conditional knowledge (full-execution truth):** on a self-built
dual-extruder system with rewritten firmware, operated attentively:
*retraction is mostly a trap with flexibles* — soft filament acts as a spring
and re-primes as delayed ooze. The working protocol controls ooze
**thermally** (cap the parked nozzle, actively cool it while parked, reheat
just-in-time, move to position fast), and catches transition mess on a
sacrificial **full-height, multi-wall draft shield surrounding the part**,
where every color change's first and last moves land — prime tower, string
trap, and thermal envelope in one structure.

The expert protocol flatly contradicts the consensus advice *as text*. As
scoped knowledge, there is no contradiction at all: one is a default, the
other is an exception whose scope is a specific hardware stack plus a
practiced operator. Both belong in the lattice. Neither may impersonate the
other.

Note the further split *inside* the expert knowledge: the **principle**
("ooze control for flexibles is thermal, not retraction-based") transfers
broadly and can inform anyone's thinking. The **procedure** (the exact
cap/cool/reheat choreography and its timings) is bound to one machine and one
set of hands. Principles travel; procedures are context-bound. The memory
should know which it is holding.

## 4. The constitution

The fix reused machinery the lattice already had — the same
default-plus-scoped-exception structure used for ordinary contextual facts —
and applied it to *authority itself*. Five rules:

1. **Provenance is first-class.** Every fact carries its tier (operator
   primary / curated import / web extract) in a dedicated, searchable field —
   never inferable-only from prose.

2. **Conditions travel with facts.** Expert-conditional facts carry an
   explicit scope (`requires=`: hardware class, tuning state, attention
   level). A fact that has lost its conditions is treated as damaged, not as
   general.

3. **Principle/procedure split.** Where expert knowledge can be decomposed,
   the transferable principle and the context-bound procedure are banked as
   separate facts, cross-linked, with scope on the procedure.

4. **Scoped authority.** Operator-primary wins on conflict **about the
   operator's own context** — their machines, their materials, their
   practice. When the system advises anyone else, it must surface **both
   paths**: the robust consensus default *and* the expert path with its
   requirements. Silent substitution in either direction is forbidden.

5. **The consensus default is preserved, not defeated.** The
   median-execution recommendation is banked alongside the expert exception
   and explicitly marked as non-conflicting — differently scoped answers to
   different questions.

## 5. Implementation and self-enforcement

Two properties of the implementation are worth recording, because they say
something about operating a live memory system:

**The system superseded its own hours-old facts.** The original absolute
authority rule and the first unscoped expert facts had been banked earlier
the same day. When the constitution landed, they were not edited in place —
they were formally superseded by scoped replacements, with history retained.
Revising the *shape* of beliefs, not merely their content, and doing it
without sentimentality about fresh work, is the behavior you want from a
system that will hold beliefs for years.

**The rule was made self-perpetuating.** Meta-facts were added requiring all
*future* operator-primary banking to arrive with scope and a
principle/procedure split, and the system's retrieval-quiz ritual (situation
probes against the lattice, scored honestly, with reinforcement for facts
that fire under pressure) tests for compliance. A constitution that depends
on everyone remembering it is a suggestion; this one is enforced at write
time and audited at recall time.

One additional detail from the same day's implementation: the system scoped a
fact we had *not* flagged — a hardware recommendation that was phrased as
universal but is actually audience-conditional — entirely unprompted. A rule
that generalizes correctly on first contact with unflagged cases is a rule
that was understood, not merely applied.

## 6. Why this matters beyond one workshop

Any long-lived agent memory that ingests both expert testimony and public
consensus will hit this problem. The failure is quiet: the system's answers
remain fluent, sourced, and individually true, while becoming wrong *for the
person asking*. Common shapes of the failure:

- Expert shortcuts served to beginners as defaults (harm through overreach).
- Beginner-safe defaults served to the expert as corrections (value
  destroyed through regression to the mean).
- A community's hard-won conditional lore flattened into "tips" that fail at
  median execution and discredit their source.

The remedy is not better ranking or a stronger authority hierarchy — every
absolute hierarchy reproduces the problem for whoever isn't at its center.
The remedy is making **scope a property of knowledge itself**, so the store
can answer the only question that matters at recall time: *true for whom,
under what conditions?*

A memory that knows what kind of true its facts are is doing something
neither textbooks nor forums manage. It lets the expert and the median
coexist without either being wrong — which is, not coincidentally, the only
condition under which an expert can safely pour a lifetime of conditional
knowledge into a system that will one day advise other people.

---

*Series context: this note extends the lattice's standing design principle
that abstraction is contextualization, not erasure — defaults update while
exceptions survive, scoped. The present document applies that same machinery
one level up: to authority, provenance, and the transferability of expertise
itself.*
