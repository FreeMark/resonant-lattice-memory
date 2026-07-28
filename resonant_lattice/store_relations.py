"""store_relations.py - RelationsMixin: Phase-5a (subject, relation, object)
triple extraction, storage, and bound-HRR encoding.

Mixed into LatticeStore; relies on the composite for self._conn/_lock, self.hrr_dim,
self._extract_entities, and self._clean_llm_json (AbstractionMixin staticmethod).

Design stance (anti-fabrication / precision first - garbage triples are the named
Phase-5 risk, and 5b/5c consume these): a deterministic, ENTITY-GROUNDED pattern
extractor is the default. A triple's confidence is boosted when its subject/object
resolve to recognized entities and penalized when neither does, so the default
relation_min_confidence gate keeps ungrounded noise out. An optional LLM pass
(default OFF) can augment recall; its output runs through the SAME grounding +
confidence gate. Inference is NOT done here - these are extracted, not derived.
"""

import json
import logging
import re
import urllib.request
from reason_gate import reason_slot
from typing import Dict, List, Optional, Tuple

from store_common import hrr, _HRR_AVAILABLE

logger = logging.getLogger(__name__)


# Edge stopwords for trimming/normalizing an extracted argument span. Kept local
# (not imported from entity_extractor, which may be absent) and deliberately small:
# articles, possessive determiners, common prepositions/pronouns that show up at
# the boundary of a captured noun phrase.
_ARG_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "their", "our", "its",
    "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "at",
    "and", "or", "but", "then", "very", "really", "just", "also",
})

# Determiners stripped from the FRONT of an argument before entity matching.
_LEADING_DETERMINERS = (
    "the ", "a ", "an ", "my ", "your ", "his ", "her ",
    "their ", "our ", "its ", "this ", "that ",
)

# Clause connectors - an argument span is cut here so a captured phrase never
# straddles a conjunction ("acme and bob" → the clause nearest the verb).
_CONNECTORS = re.compile(r"\b(?:and|or|but|then)\b|[;,]", re.IGNORECASE)

# A connector immediately BEFORE the verb ("... and lives in ...") signals an
# elided subject (coordination): the real subject belongs to the prior clause.
# A connector earlier in the span ("... and Alice works at ...") is fine - a real
# subject still sits between it and the verb. Only the trailing case is dropped.
_TRAILING_CONNECTOR = re.compile(r"(?:\b(?:and|or|but|then)\b|[;,])\s*$", re.IGNORECASE)

# One captured argument: 1-4 word tokens (word chars, apostrophes, hyphens, dots).
_ARG = r"[\w][\w'\-.]*(?:\s+[\w'\-.]+){0,3}"


def _rel(verb: str) -> "re.Pattern":
    """Compile a subject-VERB-object pattern with named subj/obj groups."""
    return re.compile(
        rf"(?P<subj>{_ARG})\s+(?:{verb})\s+(?P<obj>{_ARG})",
        re.IGNORECASE,
    )


# Relation lexicon, HIGH PRIORITY FIRST. Specific patterns (named, located_in,
# part_of) precede the generic is_a so "X is a member of Y" is read as part_of,
# not (X, is_a, member). Overlapping matches are suppressed by span-claiming in
# extract_triples. base = prior confidence; entity grounding adjusts from there.
_REL_PATTERNS: List[Tuple["re.Pattern", str, float]] = [
    # possessive name: "Alice's name is Maya"
    (re.compile(rf"(?P<subj>{_ARG})'s\s+name\s+is\s+(?P<obj>{_ARG})", re.IGNORECASE),
     "named", 0.65),
    (_rel(r"is\s+(?:named|called)|are\s+(?:named|called)"), "named", 0.6),

    (_rel(r"works?\s+(?:at|for)"), "works_at", 0.6),
    (_rel(r"works?\s+on"), "works_on", 0.55),

    (_rel(r"lives?\s+in|resides?\s+in|(?:is|are)\s+based\s+in|based\s+in"),
     "lives_in", 0.6),
    (_rel(r"(?:is|are)\s+located\s+in|located\s+in|(?:is|are)\s+situated\s+in"),
     "located_in", 0.6),
    (_rel(r"(?:was|were)\s+born\s+in|born\s+in"), "born_in", 0.6),

    (_rel(r"(?:is|are)\s+part\s+of|belongs?\s+to|"
          r"(?:is|are)\s+a\s+member\s+of|member\s+of"), "part_of", 0.55),

    (_rel(r"prefers?|likes?|loves?|enjoys?|favou?rs?"), "prefers", 0.55),
    (_rel(r"dislikes?|hates?|avoids?"), "dislikes", 0.55),
    (_rel(r"created|built|made|wrote|founded|developed|designed|authored"),
     "created", 0.55),
    (_rel(r"uses?|runs?|relies\s+on"), "uses", 0.5),
    (_rel(r"owns?|possess(?:es)?|(?:has|have)"), "has", 0.5),

    # generic copula LAST: "X is a/an/the Y"
    (_rel(r"(?:is|are)\s+(?:an?|the)"), "is_a", 0.5),
]

_MAX_TRIPLES_PER_FACT = 8

# Question/aux words that may be capitalized in a query but are never anchors.
_QUERY_NON_ANCHORS = frozenset({
    "who", "what", "which", "where", "whose", "whom", "when", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "the", "a", "an",
})

# Phase 5c: relations whose composition with ITSELF is still meaningful, so a
# same-relation chain may be labelled with that transitive closure (Seattle
# located_in WA, WA located_in USA => located_in). Mixed-relation chains are NEVER
# given a composed relation name - they are surfaced as a path only.
_TRANSITIVE_RELATIONS = frozenset({"located_in", "part_of", "lives_in"})

# Phase 5b free-query relation detection: map a verb keyword in a question to a
# canonical relation (the same vocabulary _REL_PATTERNS produces). HIGH PRIORITY
# FIRST - "works on" is checked before the generic "works" so it wins.
_QUERY_REL_KEYWORDS: List[Tuple["re.Pattern", str]] = [
    (re.compile(r"\bworks?\s+on\b", re.IGNORECASE), "works_on"),
    (re.compile(r"\b(?:works?|working|employed)\b", re.IGNORECASE), "works_at"),
    (re.compile(r"\b(?:lives?|living|reside[sd]?|based)\b", re.IGNORECASE), "lives_in"),
    (re.compile(r"\b(?:located|situated|location)\b", re.IGNORECASE), "located_in"),
    (re.compile(r"\bborn\b", re.IGNORECASE), "born_in"),
    (re.compile(r"\b(?:named|name|called)\b", re.IGNORECASE), "named"),
    (re.compile(r"\b(?:member|belongs?|part)\b", re.IGNORECASE), "part_of"),
    (re.compile(r"\b(?:prefers?|likes?|loves?|enjoys?|favou?rs?)\b", re.IGNORECASE), "prefers"),
    (re.compile(r"\b(?:dislikes?|hates?|avoids?)\b", re.IGNORECASE), "dislikes"),
    (re.compile(r"\b(?:created?|built|made|wrote|founded|developed|designed|authored)\b",
                re.IGNORECASE), "created"),
    (re.compile(r"\b(?:uses?|using|runs?|relies)\b", re.IGNORECASE), "uses"),
    (re.compile(r"\b(?:owns?|possess(?:es)?|have|has)\b", re.IGNORECASE), "has"),
]


def _resolve_arg(span: str, ent_set: set, side: str,
                 is_value: bool = False) -> Tuple[Optional[str], bool]:
    """Normalize one captured argument and try to ground it to a known entity.

    `side` is "subj" (keep the clause nearest the verb → rightmost segment) or
    "obj" (keep the leftmost segment), so a phrase never straddles a conjunction.
    Returns (normalized_arg, grounded). grounded=True when the span resolves to a
    recognized entity (highest precision). An ungrounded span survives only if it
    is a short (≤3-word) clean noun phrase; longer ungrounded spans return None so
    the triple is dropped. Lowercased throughout, matching entity-graph keys.

    ``is_value`` marks the OBJECT of an ATTRIBUTE predicate (reference_interval,
    dosed_at, presents_as, stage_defined_by ...), whose object is a measurement or a
    phrase rather than a graph node. Node normalisation destroys those three separate
    ways, all measured on a live re-extraction:

      * entity grounding returns the matched ENTITY, not the span, so "creatinine
        below 125 micromol/L" collapses to "creatinine" -- every stage_defined_by
        object arrived as a bare analyte name carrying no number;
      * connector splitting cuts at "and", so presents_as "drinking and urinating
        noticeably more than usual" becomes "drinking";
      * the >3-word ceiling rejects most intervals outright.

    That is the worst failure mode available here: a graph that looks populated and
    holds no values. For a value the span is kept verbatim apart from determiner and
    punctuation trimming, and returned UNGROUNDED -- it is not a node and must not
    claim to be one. Grounding of the SUBJECT is untouched, so relation_require_entity
    still refuses fragment subjects.
    """
    if not span:
        return None, False
    s = span.strip().lower()

    if not is_value:
        # Cut at clause connectors, keep the segment nearest the verb.
        parts = [p.strip() for p in _CONNECTORS.split(s) if p and p.strip()]
        if parts:
            s = parts[-1] if side == "subj" else parts[0]

    # Strip a leading determiner/possessive.
    for det in _LEADING_DETERMINERS:
        if s.startswith(det):
            s = s[len(det):]
            break

    s = s.strip().strip("\"'.,;:!?()[]{}").strip()
    if len(s) < 2:
        return None, False

    if is_value:
        return s, False

    # Entity grounding: longest entity that is the whole span or a whole-word
    # substring of it (so "i think alice" grounds to the entity "alice").
    grounded = None
    for e in ent_set:
        if not e:
            continue
        if e == s or re.search(rf"\b{re.escape(e)}\b", s):
            if grounded is None or len(e) > len(grounded):
                grounded = e
    if grounded is not None:
        return grounded, True

    # Ungrounded: only accept a short, clean noun phrase. Trim stopword edges.
    tokens = s.split()
    while tokens and tokens[0] in _ARG_STOPWORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in _ARG_STOPWORDS:
        tokens.pop()
    if not tokens or len(tokens) > 3:
        return None, False
    return " ".join(tokens), False


# An alias replaces the WHOLE argument, so a short acronym that also PREFIXES
# composite identifiers silently destroys them: with iris -> "international renal
# interest society", the span "iris ckd stage 2" became the organisation and the stage
# was lost; "usg >1.030" would likewise become "urine specific gravity" and drop the
# threshold. Requiring the alias to cover most of the span keeps the intended
# collapses ("46" -> "node .46", "nemotron-3-super:cloud" -> "nemotron", "usg" ->
# "urine specific gravity") while leaving composite spans alone. A whole-VALUE match
# always wins regardless of length.
_ALIAS_MIN_COVERAGE = 0.6


def _alias_canon(value: Optional[str], amap: dict) -> Optional[str]:
    """Single implementation of alias canonicalisation, shared by the storage side
    (_canonicalize_triples) and the query side (_canon_term) so the two cannot drift
    apart -- if storage stops collapsing a span, a query for it must stop too."""
    if not value or not amap:
        return value
    low = value.strip().lower()
    if low in amap:
        return amap[low]
    best = None
    for surf, canonical in amap.items():
        if not surf or len(surf) < _ALIAS_MIN_COVERAGE * len(low):
            continue
        if re.search(rf"\b{re.escape(surf)}\b", low):
            if best is None or len(surf) > len(best[0]):
                best = (surf, canonical)      # longest alias wins, not dict order
    return best[1] if best else value


def _canon_term(term: Optional[str], amap: dict) -> Optional[str]:
    """Map a QUERY term to its canonical node via the alias map (whole-value or whole-word
    match), so the query side matches the aliased stored graph (asking about 'the node' finds
    'node .46'). Returns the term unchanged when no alias applies. Mirrors the storage-side
    canonicalization in _canonicalize_triples -- both delegate to _alias_canon so the two
    sides cannot diverge."""
    return _alias_canon(term, amap)


class RelationsMixin:

    # ====================== TRIPLE EXTRACTION ======================
    def extract_triples(self, content: str, entities: Optional[List[str]] = None,
                        vocabulary: Optional[List[str]] = None) -> List[Dict]:
        """Deterministic, entity-grounded (s, relation, o) extraction from content.

        LLM-free. Applies the relation lexicon high-priority-first with span-claiming
        (a region matched by a specific relation is not re-parsed by the generic
        is_a). Returns a list of {subject, relation, object, confidence} dicts,
        deduped and capped. Confidence reflects entity grounding; the caller gates
        on relation_min_confidence. Read-only / pure - does not touch the DB.

        When ``vocabulary`` is given (a closed relation set for the agent's domain),
        only patterns whose relation is IN the vocabulary are applied - so a domain
        that supplies ["uses","part_of","runs_on",...] keeps the deterministic
        uses/part_of matches but drops the generic is_a/has noise (the
        "grok is_a bottleneck" / "memory has tidied" misparses). Empty/None = legacy
        behaviour (all patterns).
        """
        if not content or not content.strip():
            return []
        text = content.strip()
        if entities is None:
            try:
                entities = self._extract_entities(text)
            except Exception:
                entities = []
        ent_set = {e.lower() for e in (entities or []) if e}
        vocab_set = {v.strip().lower() for v in vocabulary if v and v.strip()} if vocabulary else None

        claimed: List[Tuple[int, int]] = []
        seen: set = set()
        out: List[Dict] = []

        for pattern, relation, base in _REL_PATTERNS:
            if vocab_set is not None and relation not in vocab_set:
                continue  # domain denoise: skip patterns outside the closed vocabulary
            for m in pattern.finditer(text):
                # Claim only the VERB region (between subject and object), not the
                # whole match: this blocks the generic is_a from re-parsing a span a
                # specific relation already consumed, WITHOUT suppressing a genuinely
                # adjacent triple whose greedy object span happened to overrun into
                # the next clause (e.g. "Bob lives in Paris and Alice works at Acme").
                start, end = m.end("subj"), m.start("obj")
                if start > end:
                    start, end = end, start
                if any(not (end <= c0 or start >= c1) for c0, c1 in claimed):
                    continue
                # Elided-subject guard: in "X verb1 Y and verb2 Z" the second verb's
                # real subject (X) is elided, so the captured subject span ENDS with
                # the connector (the verb directly follows "and"). We can't recover X,
                # and emitting (Y, verb2, Z) would be a CONFIDENT WRONG triple - drop
                # it (precision over recall; anti-fabrication).
                if _TRAILING_CONNECTOR.search(m.group("subj")):
                    continue
                subj, subj_g = _resolve_arg(m.group("subj"), ent_set, "subj")
                obj, obj_g = _resolve_arg(m.group("obj"), ent_set, "obj")
                if not subj or not obj or subj == obj:
                    continue
                conf = base + (0.15 if subj_g else 0.0) + (0.15 if obj_g else 0.0)
                if not subj_g and not obj_g:
                    conf *= 0.6   # ungrounded → push below the default gate
                conf = round(min(conf, 1.0), 3)
                key = (subj, relation, obj)
                if key in seen:
                    continue
                seen.add(key)
                claimed.append((start, end))
                out.append({"subject": subj, "relation": relation,
                            "object": obj, "confidence": conf})
                if len(out) >= _MAX_TRIPLES_PER_FACT:
                    return out
        return out

    # The KINDS of thing a graph is made of. This list used to be hardcoded into the
    # prompt below, which silently made the closed-vocabulary mechanism
    # infrastructure-only: a clinical lattice instructed to extract "components,
    # machines, services, settings, files, versions" returns [] on "Urine must be
    # cultured prior to antimicrobic administration", because the model correctly
    # reports that the note contains no machines. Measured on a 6,983-fact veterinary
    # corpus, 2 of 3 sampled facts came back empty under this wording while the
    # vocabulary itself was being enforced perfectly. Kept as the DEFAULT so existing
    # operational profiles build a byte-identical prompt; override per profile with
    # relation_subject_kinds.
    _DEFAULT_SUBJECT_KINDS = ("components, machines, models, services, tools, "
                              "settings, files, versions, people, places")

    @staticmethod
    def _build_relation_prompt(vocabulary: List[str], examples: Optional[str],
                               subject_kinds: Optional[str] = None) -> str:
        """Schema-constrained slot-filling prompt built from a closed relation vocabulary.

        Turns triple extraction from open generation ("invent a snake_case relation")
        into classification ("pick a relation from THIS list, name real things, or
        return []") - the task a small local model does reliably. Optional domain
        ``examples`` are appended verbatim (few-shot). Validated at scale to yield a
        near-100%-canonical, recurring relation set.

        ``subject_kinds`` names the kinds of thing THIS domain's graph is built from
        (see _DEFAULT_SUBJECT_KINDS for why it is a parameter and not a constant). The
        vocabulary, the examples and this phrase are the three places the domain
        enters; everything else in the prompt is domain-neutral."""
        vocab_line = ", ".join(vocabulary)
        ex = ("\n" + examples) if examples else ""
        kinds = (subject_kinds or "").strip() or RelationsMixin._DEFAULT_SUBJECT_KINDS
        return (
            "You are building a KNOWLEDGE GRAPH from the note "
            f"below. Extract ONLY relationships between NAMED things ({kinds}).\n"
            f"Use ONLY these relation types (choose the closest; if none fit, omit the triple): "
            f"{vocab_line}.\n"
            "Rules:\n"
            "- subject and object must be NAMED things copied from the text. NEVER use pronouns, "
            "whole sentences, feelings, or vague nouns ('it', 'this', 'the system', 'my job').\n"
            "- If the note has no two named things in one of these relations (it is introspection, "
            "a plan, or a feeling), return [].\n"
            "- Only what is literally stated; never infer.\n"
            'Output ONLY a JSON array of {"subject","relation","object"} with relation from the '
            f"list, or [].{ex}"
        )

    def _llm_extract_triples(self, content: str, reason_model: str,
                             ollama_endpoint: str,
                             entities: Optional[List[str]] = None,
                             prompt: Optional[str] = None,
                             vocabulary: Optional[List[str]] = None,
                             examples: Optional[str] = None) -> List[Dict]:
        """Optional LLM augmentation (default OFF). Conservative + non-fatal.

        Asks the reasoning model for explicit (subject, relation, object) triples,
        then runs each through the SAME grounding/normalization as the deterministic
        path so the LLM cannot inject ungrounded long spans. Returns [] on any error
        (network, parse) so a failed pass never blocks consolidation.

        When ``vocabulary`` is given, the prompt is a schema-constrained slot-filler
        over that closed set (unless an explicit ``prompt`` overrides it) AND every
        extracted relation is validated to be IN the vocabulary - off-list relations
        (the free-form run-on noise a generic prompt produces) are dropped. This is
        what makes relations RECUR (a traversable graph) instead of each being a
        one-off. Empty/None vocabulary = legacy free-form behaviour.
        """
        if not content or not content.strip():
            return []
        if entities is None:
            try:
                entities = self._extract_entities(content)
            except Exception:
                entities = []
        ent_set = {e.lower() for e in (entities or []) if e}
        vocab_set = {v.strip().lower() for v in vocabulary if v and v.strip()} if vocabulary else None

        if prompt:
            base_prompt = prompt                       # explicit override wins
        elif vocab_set:
            base_prompt = self._build_relation_prompt(
                list(vocabulary), examples,
                getattr(self, "relation_subject_kinds", None))
        else:
            base_prompt = (
                "Extract explicit (subject, relation, object) triples STATED in the text "
                "below. Only facts literally present - never infer or add world knowledge. "
                "relation is a short snake_case verb phrase. Output ONLY a JSON array of "
                'objects with keys "subject", "relation", "object", or [] if none.'
            )
        final_prompt = f"{base_prompt}\n\nTEXT:\n{content}\n\nJSON OUTPUT:"
        try:
            # BOUND THE GENERATION. A triple list is inherently short -- at most
            # _MAX_TRIPLES_PER_FACT (8) objects, ~224 tokens -- but with no cap the
            # endpoint's own default applies, and a model that occasionally fails to
            # stop runs to it. Measured against a vLLM shim whose default was 8192:
            # typical calls emitted 41 tokens while the tail hit the 300s timeout
            # below, which this method then swallowed into a non-fatal [] -- a lost
            # extraction indistinguishable from "this fact states no relations".
            # 0/unset keeps the endpoint default (previous behaviour).
            _np = getattr(self, "relation_num_predict", 0)
            _opts = {"temperature": 0.1}
            try:
                if int(_np) > 0:
                    _opts["num_predict"] = int(_np)
            except (TypeError, ValueError):
                pass
            payload = {"model": reason_model, "prompt": final_prompt,
                       "stream": False, "think": False, "options": _opts}
            req = urllib.request.Request(
                f"{ollama_endpoint}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with reason_slot(), urllib.request.urlopen(req, timeout=300.0) as response:
                raw = json.loads(response.read().decode("utf-8")).get("response", "[]")
            cleaned = self._clean_llm_json(raw)
            start_idx, end_idx = cleaned.find("["), cleaned.rfind("]")
            parsed = json.loads(cleaned[start_idx:end_idx + 1]
                                if start_idx != -1 and end_idx != -1 else cleaned)
        except Exception as e:
            logger.debug("LLM triple extraction failed (non-fatal): %s", e)
            return []
        if not isinstance(parsed, list):
            return []

        out: List[Dict] = []
        seen: set = set()
        for item in parsed:
            # POSITIONAL FORM TOLERATED. A model given few-shot examples written as
            # [["subject","relation","object"], ...] returns that shape faithfully,
            # and dict-only parsing discarded EVERY triple without a trace. Measured:
            # the same prompt returned dicts while reasoning was on and arrays once it
            # was off, so this silently halved to nothing depending on an unrelated
            # serving flag. Accept both; a 3-element sequence is unambiguous.
            if isinstance(item, (list, tuple)) and len(item) == 3:
                item = {"subject": item[0], "relation": item[1], "object": item[2]}
            if not isinstance(item, dict):
                continue
            rel_raw = item.get("relation")
            if not isinstance(rel_raw, str):
                continue
            relation = re.sub(r"[^a-z_]+", "_", rel_raw.strip().lower()).strip("_")
            if not relation:
                continue
            if vocab_set is not None and relation not in vocab_set:
                continue  # drop off-vocabulary relations (the free-form one-off noise)
            # An attribute predicate's object is a VALUE, so it must not be collapsed
            # to an entity, split at a conjunction, or length-capped. The subject is
            # still normalised as a node.
            _is_value = relation in self._attribute_relations()
            subj, subj_g = _resolve_arg(str(item.get("subject", "")), ent_set, "subj")
            obj, obj_g = _resolve_arg(str(item.get("object", "")), ent_set, "obj",
                                      is_value=_is_value)
            if not subj or not obj or subj == obj:
                continue
            conf = 0.7 + (0.1 if subj_g else 0.0) + (0.1 if obj_g else 0.0)
            if not subj_g and not obj_g:
                conf *= 0.6
            conf = round(min(conf, 1.0), 3)
            key = (subj, relation, obj)
            if key in seen:
                continue
            seen.add(key)
            out.append({"subject": subj, "relation": relation,
                        "object": obj, "confidence": conf})
            if len(out) >= _MAX_TRIPLES_PER_FACT:
                break
        return out

    # ====================== STORAGE ======================
    def _encode_triple_blob(self, subject: str, relation: str, object_: str) -> Optional[bytes]:
        """Bound-HRR encoding of a triple as storable bytes (None if HRR is off)."""
        if not _HRR_AVAILABLE or hrr is None:
            return None
        try:
            vec = hrr.encode_triple(subject, relation, object_, dim=self.hrr_dim)
            return hrr.phases_to_bytes(vec)
        except Exception as e:
            logger.debug("Triple HRR encoding failed (non-fatal): %s", e)
            return None

    def store_fact_relations(self, fact_id: int, triples: List[Dict],
                             min_confidence: float = 0.5) -> int:
        """Persist gated triples for a fact; returns the number inserted.

        Each triple at/above min_confidence is encoded as a bound HRR vector and
        written with INSERT OR IGNORE against UNIQUE(fact_id, subject, relation,
        object), so re-extraction of the same fact is idempotent. Lock-guarded.
        """
        if fact_id is None or fact_id < 0 or not triples:
            return 0
        rows = []
        for t in triples:
            if t.get("confidence", 0.0) < min_confidence:
                continue
            subj, rel, obj = t.get("subject"), t.get("relation"), t.get("object")
            if not subj or not rel or not obj:
                continue
            rows.append((fact_id, subj, rel, obj, float(t["confidence"]),
                         self._encode_triple_blob(subj, rel, obj)))
        if not rows:
            return 0
        with self._lock:
            cur = self._conn.executemany(
                """
                INSERT OR IGNORE INTO fact_relations
                    (fact_id, subject, relation, object, confidence, hrr_vector)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
            return cur.rowcount or 0

    # Relations whose OBJECT is a value/attribute (a number, a flag, a path), not a
    # graph node - so the object is NOT required to resolve to a known entity.
    # Relations whose OBJECT is a VALUE rather than a graph node, so strict entity
    # binding must not require it to be a known entity. The two built-ins are
    # operational; a domain's closed vocabulary is typically mostly attributes
    # (reference_interval -> a range, dosed_at -> a dose, presents_as -> a sign
    # phrase), and with relation_require_entity on it would lose all of them. Extend
    # per profile with relation_attribute_predicates rather than editing this set --
    # the closed-vocabulary mechanism is incomplete without a way for a domain to say
    # WHICH of its predicates take a value.
    _ATTRIBUTE_RELATIONS = frozenset({"set_to", "produces"})

    def _attribute_relations(self) -> frozenset:
        extra = getattr(self, "relation_attribute_predicates", None) or ()
        return self._ATTRIBUTE_RELATIONS | {
            str(p).strip().lower() for p in extra if str(p).strip()}

    def _canonicalize_triples(self, triples: List[Dict], content: str,
                              entities: Optional[List[str]],
                              aliases: Optional[Dict[str, str]],
                              require_entity: bool) -> List[Dict]:
        """Phase-2 entity binding: unify node names to canonical forms and (optionally)
        drop triples whose graph args are not named entities.

        - ``aliases`` maps a surface form to its canonical node (e.g. '46' / '.46' /
          'node' -> 'node .46'; 'nemotron-3-super:cloud' -> 'nemotron'). Applied to
          subject and object after extraction, so the SAME real thing becomes ONE
          node - which is what lets triples share nodes and chains form. Keys are
          matched case-insensitively, as a whole value OR a whole-word substring.
        - ``require_entity`` (strict binding): a component<->component triple is kept
          only if BOTH args resolve to a known entity (from the fact's entity set +
          the alias canonicals); attribute relations (set_to/produces) exempt the
          object. Drops fragment/pronoun noise ('/consolidation', 'mid-session') at
          the cost of some recall. Re-dedups after aliasing. Pure; no DB access."""
        if not aliases and not require_entity:
            return triples
        amap = {str(k).strip().lower(): str(v).strip() for k, v in (aliases or {}).items() if k and v}

        def canon(arg: str) -> str:
            # whole-value, else a whole-word alias that covers most of the span
            # (see _ALIAS_MIN_COVERAGE for why partial coverage must NOT replace)
            return _alias_canon((arg or "").strip(), amap) or (arg or "").strip()

        for t in triples:
            t["subject"] = canon(t["subject"])
            t["object"] = canon(t["object"])

        if require_entity:
            ent = {e.strip().lower() for e in (entities if entities is not None
                                               else self._extract_entities(content)) if e}
            ent |= {v.strip().lower() for v in amap.values()}       # alias canonicals count as entities

            def grounded(arg: str) -> bool:
                a = (arg or "").strip().lower()
                return bool(a) and (a in ent or any(
                    e == a or re.search(rf"\b{re.escape(e)}\b", a) for e in ent))
            attrs = self._attribute_relations()
            triples = [t for t in triples if grounded(t["subject"]) and (
                t["relation"] in attrs or grounded(t["object"]))]

        seen, out = set(), []                              # re-dedup (aliasing can merge)
        for t in triples:
            k = (t["subject"], t["relation"], t["object"])
            if t["subject"] and t["object"] and t["subject"] != t["object"] and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def extract_and_store_relations(self, fact_id: int, content: str,
                                    entities: Optional[List[str]] = None,
                                    min_confidence: float = 0.5,
                                    reason_model: Optional[str] = None,
                                    ollama_endpoint: Optional[str] = None,
                                    use_llm: bool = False,
                                    llm_prompt: Optional[str] = None,
                                    vocabulary: Optional[List[str]] = None,
                                    examples: Optional[str] = None,
                                    aliases: Optional[Dict[str, str]] = None,
                                    require_entity: bool = False) -> int:
        """Extract triples from a fact's content and persist the confident ones.

        Deterministic extraction always runs; the optional LLM pass (use_llm) only
        runs when a model + endpoint are supplied, and its triples are merged
        (deterministic taking precedence on a key collision). Returns inserted count.

        ``vocabulary`` (a closed relation set for the agent's domain) constrains BOTH
        paths to relations in the set: the deterministic path drops out-of-vocab
        patterns (is_a/has noise) and the LLM path becomes a validated slot-filler.
        ``examples`` are domain few-shots for the LLM prompt. ``aliases`` +
        ``require_entity`` are the Phase-2 entity-binding step (node canonicalization
        + optional strict grounding). All default None/off (legacy behaviour).
        """
        triples = self.extract_triples(content, entities, vocabulary=vocabulary)
        if use_llm and reason_model and ollama_endpoint:
            have = {(t["subject"], t["relation"], t["object"]) for t in triples}
            for t in self._llm_extract_triples(content, reason_model, ollama_endpoint,
                                               entities=entities, prompt=llm_prompt,
                                               vocabulary=vocabulary, examples=examples):
                if (t["subject"], t["relation"], t["object"]) not in have:
                    triples.append(t)
        triples = self._canonicalize_triples(triples, content, entities, aliases, require_entity)
        return self.store_fact_relations(fact_id, triples, min_confidence)

    def extract_transcript_relations(self, text: str, candidate_facts: List[Tuple[int, str]],
                                     min_confidence: float = 0.5,
                                     reason_model: Optional[str] = None,
                                     ollama_endpoint: Optional[str] = None,
                                     vocabulary: Optional[List[str]] = None,
                                     examples: Optional[str] = None,
                                     aliases: Optional[Dict[str, str]] = None,
                                     llm_prompt: Optional[str] = None) -> int:
        """Phase-4: extract relations from a RAW TRANSCRIPT window (before consolidation compresses
        it) and attach each to the born fact it best matches.

        The transcript states conversational DEPENDENCY / DECISION / USAGE relationships that
        per-fact extraction loses ('grok uses rlm_pin', 'infer depends_on relational', 'we set X
        because Y'). It is also noisier (disfluent speech fragments), so this path forces STRICT
        entity binding (require_entity=True) -- the same strict mode that is too aggressive for
        clean facts is exactly right here, dropping fragments like 'll' / 'can' / 'make it simpler'.

        ``candidate_facts`` = [(fact_id, content)] the facts BORN from this window; each triple is
        attached to the one whose content best overlaps its subject+object (the evidence anchor +
        the fact_relations FK target), so relations tie to a plausible source fact and de-duplicate
        (each triple lands once, not once per born fact). Returns the count stored. Non-fatal caller
        expected. Merges with, does not replace, the per-fact relations."""
        if not text or not text.strip() or not candidate_facts:
            return 0
        entities = self._extract_entities(text)
        triples = self.extract_triples(text, entities, vocabulary=vocabulary)
        if reason_model and ollama_endpoint:
            have = {(t["subject"], t["relation"], t["object"]) for t in triples}
            for t in self._llm_extract_triples(text, reason_model, ollama_endpoint,
                                               entities=entities, prompt=llm_prompt,
                                               vocabulary=vocabulary, examples=examples):
                if (t["subject"], t["relation"], t["object"]) not in have:
                    triples.append(t)
        triples = self._canonicalize_triples(triples, text, entities, aliases, require_entity=True)
        if not triples:
            return 0

        def _best_fact(triple: Dict) -> int:
            toks = set((triple["subject"] + " " + triple["object"]).lower().split())
            best_id, best_score = candidate_facts[0][0], -1
            for fid, content in candidate_facts:
                score = len(toks & set((content or "").lower().split()))
                if score > best_score:
                    best_id, best_score = fid, score
            return best_id

        by_fact: Dict[int, List[Dict]] = {}
        for t in triples:
            by_fact.setdefault(_best_fact(t), []).append(t)
        return sum(self.store_fact_relations(fid, ts, min_confidence) for fid, ts in by_fact.items())

    # ====================== READS (substrate / inspection) ======================
    def get_fact_relations(self, fact_id: int) -> List[Dict]:
        """All triples extracted from one fact, strongest first. Read-only."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT subject, relation, object, confidence
                FROM fact_relations WHERE fact_id = ?
                ORDER BY confidence DESC, relation ASC
                """,
                (fact_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_relations(self, subject: Optional[str] = None,
                      object: Optional[str] = None,
                      relation: Optional[str] = None,
                      min_confidence: float = 0.0,
                      include_superseded: bool = False,
                      limit: int = 100) -> List[Dict]:
        """Exact-match triple lookup over the graph (Phase-5a substrate read).

        Plain SQL filter on the normalized subject/object/relation columns - this
        is the index-backed primitive Phase 5b builds HRR-fuzzy recall and bounded
        inference on top of. By default joins to live facts (superseded belief-
        history excluded). Read-only.
        """
        clauses = ["r.confidence >= ?"]
        params: list = [min_confidence]
        if subject is not None:
            clauses.append("r.subject = ?")
            params.append(subject.strip().lower())
        if object is not None:
            clauses.append("r.object = ?")
            params.append(object.strip().lower())
        if relation is not None:
            clauses.append("r.relation = ?")
            params.append(relation.strip().lower())
        if not include_superseded:
            clauses.append("f.tier != 'superseded'")
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT r.fact_id, r.subject, r.relation, r.object, r.confidence
                FROM fact_relations r
                JOIN semantic_facts f ON f.id = r.fact_id
                WHERE {' AND '.join(clauses)}
                ORDER BY r.confidence DESC, r.relation ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    # ====================== RELATIONAL RECALL (Phase 5b) ======================
    def relational_recall(self, subject: Optional[str] = None,
                          relation: Optional[str] = None,
                          object: Optional[str] = None,
                          anchors: Optional[List[str]] = None,
                          query: Optional[str] = None,
                          max_results: int = 10,
                          min_confidence: float = 0.0,
                          hrr_floor: float = 0.4,
                          include_superseded: bool = False,
                          aliases: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Answer a relational query over the triple graph (Phase 5b).

        Two complementary passes, ranked together:
          • GRAPH (exact, index-backed SQL): triples satisfying every provided
            role slot (subject/relation/object) AND every role-agnostic `anchor`
            (a term that may appear as subject OR object). These rank first.
          • HRR (fuzzy, graceful fallback): a partial-binding probe
            (holographic.encode_triple_query) scores the remaining candidates by
            how much of the known structure they share, so a near-match surfaces
            when no triple satisfies ALL the constraints (e.g. asking where Free
            lives when only 'Free lives in Seattle' is stored vs a wrong city).

        A free-text `query` is parsed into (relation + anchors) when no structured
        slot is given. Read-only. Each result is labelled with match='graph'|'hrr'
        and a score, plus the source fact's content/tier (superseded excluded).
        """
        if query and not (subject or relation or object or anchors):
            relation, anchors = self._parse_relational_query(query)

        # Canonicalize the QUERY side through the same alias map as the stored graph, so a
        # query term ('node', 'the node') resolves to the stored canonical node ('node .46').
        amap = {str(k).strip().lower(): str(v).strip() for k, v in (aliases or {}).items() if k and v}
        slots = {}
        if subject:
            slots["subject"] = _canon_term(subject.strip().lower(), amap)
        if relation:
            slots["relation"] = relation.strip().lower()
        if object:
            slots["object"] = _canon_term(object.strip().lower(), amap)
        anchor_set = {_canon_term(a.strip().lower(), amap) for a in (anchors or []) if a and a.strip()}
        if not slots and not anchor_set:
            return []

        # Candidate set: any single known constraint matches (so the HRR pass can
        # see partial-structure matches the exact filter would miss). Index-backed.
        or_clauses, params = [], []
        for col, val in slots.items():
            or_clauses.append(f"r.{col} = ?")
            params.append(val)
        for a in anchor_set:
            or_clauses.append("(r.subject = ? OR r.object = ?)")
            params.extend([a, a])
        where = [f"({' OR '.join(or_clauses)})", "r.confidence >= ?"]
        params.append(min_confidence)
        if not include_superseded:
            where.append("f.tier != 'superseded'")
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT r.fact_id, r.subject, r.relation, r.object, r.confidence,
                       r.hrr_vector, f.content, f.tier
                FROM fact_relations r
                JOIN semantic_facts f ON f.id = r.fact_id
                WHERE {' AND '.join(where)}
                """,
                params,
            ).fetchall()

        probe = None
        if _HRR_AVAILABLE and hrr is not None:
            try:
                # Role-agnostic anchors are bound to BOTH roles so a triple matches
                # whether the anchor is its subject or object.
                anc_kwargs = []
                probe = hrr.encode_triple_query(
                    subject=slots.get("subject"), relation=slots.get("relation"),
                    object_=slots.get("object"), dim=self.hrr_dim,
                )
                for a in anchor_set:
                    pa = hrr.encode_triple_query(subject=a, dim=self.hrr_dim)
                    po = hrr.encode_triple_query(object_=a, dim=self.hrr_dim)
                    anc_kwargs.extend([pa, po])
                if anc_kwargs:
                    parts = ([probe] if probe is not None else []) + anc_kwargs
                    probe = hrr.bundle(*parts)
            except Exception:
                probe = None

        results = []
        for row in rows:
            full = all(row[c] == v for c, v in slots.items()) and all(
                a in (row["subject"], row["object"]) for a in anchor_set
            )
            score = None
            if probe is not None and row["hrr_vector"]:
                vec = self._phases_from_blob(row["hrr_vector"])
                if vec is not None:
                    score = round(float(hrr.similarity(probe, vec)), 3)
            if not full and (score is None or score < hrr_floor):
                continue
            results.append({
                "fact_id": row["fact_id"], "subject": row["subject"],
                "relation": row["relation"], "object": row["object"],
                "confidence": row["confidence"], "content": row["content"],
                "tier": row["tier"], "match": "graph" if full else "hrr",
                "score": score,
            })
        # Exact graph matches first (by confidence), then fuzzy by HRR score.
        results.sort(key=lambda r: (
            0 if r["match"] == "graph" else 1,
            -(r["confidence"] if r["match"] == "graph" else (r["score"] or 0.0)),
        ))
        return results[:max_results]

    def _parse_relational_query(self, query: str) -> Tuple[Optional[str], List[str]]:
        """Best-effort parse of a free-text relational question (Phase 5b).

        Returns (relation, anchors): a canonical relation if a known verb is
        detected, and the query's entities as role-agnostic anchors (we do NOT
        guess subject vs object - the recall's anchor matching and HRR ranking
        handle either orientation, so a role misguess can't drop a real answer).
        """
        if not query or not query.strip():
            return None, []
        relation = None
        for pattern, canonical in _QUERY_REL_KEYWORDS:
            if pattern.search(query):
                relation = canonical
                break
        try:
            anchors = list(self._extract_entities(query))
        except Exception:
            anchors = []
        # Fallback: proper nouns are the typical subject/object of a relational
        # question but need spaCy to be caught by the entity layer's regex (which
        # requires 2+ words). Pull them here so questions work without spaCy.
        have = {a.lower() for a in anchors}
        # (1) Multi-word capitalized SPANS ("Acme Robotics") - almost always real
        # entities regardless of position, so take them from the FULL query and
        # add the whole phrase as one anchor (matching a stored multi-word
        # subject/object). Mark the component words seen so they don't also leak in
        # as redundant single-token anchors.
        for span in re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b", query):
            low = span.lower()
            if low not in have:
                anchors.append(low)
            have.add(low)
            for w in low.split():
                have.add(w)
        # (2) Single capitalized tokens ("Free") - exclude the sentence-initial one
        # (query[1:]) and question/aux words to avoid false anchors.
        for tok in re.findall(r"\b([A-Z][a-zA-Z]{1,})\b", query[1:]):
            low = tok.lower()
            if low not in have and low not in _QUERY_NON_ANCHORS:
                anchors.append(low)
                have.add(low)
        return relation, anchors

    # ====================== TRANSITIVE INFERENCE (Phase 5c) ======================
    def infer_relations(self, subject: str, object: Optional[str] = None,
                        max_hops: int = 2, min_confidence: float = 0.0,
                        hop_decay: float = 0.6, max_results: int = 10,
                        include_superseded: bool = False,
                        aliases: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Bounded transitive inference over the triple graph (Phase 5c).

        Chains stored triples forward from `subject` (a→r1→b→r2→c …) up to
        `max_hops` and returns the multi-hop connections as **labelled inferences**.
        Example: (mark, works_at, anthropic) + (anthropic, located_in, seattle) ⇒
        an inferred connection mark→seattle, with the supporting path attached.

        CRITICAL anti-fabrication invariants (this is where discipline matters most):
          • Pure read + compute - NEVER writes to fact_relations or semantic_facts.
            Inferences are returned, never persisted, never quote_status='attested'.
          • Every result carries inferred=True, the full `path` of REAL stored
            triples it rests on, hop count, and a confidence that DECAYS per hop
            (product of hop confidences × hop_decay^(hops-1)) so a derived link is
            always weaker than a stored fact.
          • A composed `relation` name is asserted ONLY when every hop shares one
            relation that is transitively closeable (_TRANSITIVE_RELATIONS); mixed
            chains get relation=None and are surfaced as a path for the agent to
            interpret - we never invent a relation the data doesn't support.

        Only multi-hop paths (len ≥ 2 edges) are returned - a 1-edge path is just a
        stored fact (use ``relational`` for those). ``max_hops`` is the maximum path
        length in edges: 1 disables multi-hop (returns []); 2+ allows derived chains
        up to that length. If ``object`` is given, only chains terminating at it are
        returned. Cycles are prevented (a node is never revisited within a path);
        fanout and result count are bounded. Superseded history is excluded by default.
        """
        # Canonicalize the query endpoints through the alias map so 'the node' chains
        # from the stored 'node .46' (mirrors the storage + relational_recall side).
        amap = {str(k).strip().lower(): str(v).strip() for k, v in (aliases or {}).items() if k and v}
        start = _canon_term(subject.strip().lower(), amap) if subject else None
        if not start:
            return []
        # Floor at 1 (not 2): max_hops=1 is the explicit "no multi-hop" setting and
        # returns [] because the loop below only emits paths of length ≥ 2.
        max_hops = max(1, int(max_hops))
        target = _canon_term(object.strip().lower(), amap) if object else None

        def _out_edges(node: str) -> List[Dict]:
            where = ["r.subject = ?", "r.confidence >= ?"]
            params = [node, min_confidence]
            if not include_superseded:
                where.append("f.tier != 'superseded'")
            with self._lock:
                rows = self._conn.execute(
                    f"""
                    SELECT r.fact_id, r.subject, r.relation, r.object, r.confidence
                    FROM fact_relations r
                    JOIN semantic_facts f ON f.id = r.fact_id
                    WHERE {' AND '.join(where)}
                    ORDER BY r.confidence DESC
                    LIMIT 20
                    """,
                    params,
                ).fetchall()
            return [dict(r) for r in rows]

        # Breadth-first expansion, carrying each path + the set of visited nodes.
        inferences: List[Dict] = []
        frontier = [([edge], {start, edge["object"]})
                    for edge in _out_edges(start)]
        hop = 1
        while frontier and hop < max_hops and len(inferences) < max_results * 4:
            nxt = []
            for path, visited in frontier:
                tail = path[-1]["object"]
                for edge in _out_edges(tail):
                    if edge["object"] in visited:
                        continue   # cycle guard
                    new_path = path + [edge]
                    inferences.append(self._compose_inference(new_path, hop_decay))
                    nxt.append((new_path, visited | {edge["object"]}))
            frontier = nxt
            hop += 1

        if target is not None:
            inferences = [inf for inf in inferences if inf["object"] == target]
        # Strongest inferences first.
        inferences.sort(key=lambda inf: -inf["confidence"])
        return inferences[:max_results]

    @staticmethod
    def _compose_inference(path: List[Dict], hop_decay: float) -> Dict:
        """Assemble one inference result from a chain of stored triples (5c helper)."""
        n_hops = len(path)
        conf = 1.0
        for edge in path:
            conf *= float(edge.get("confidence", 0.0))
        conf *= hop_decay ** (n_hops - 1)
        rels = {edge["relation"] for edge in path}
        composed = path[0]["relation"] if (
            len(rels) == 1 and path[0]["relation"] in _TRANSITIVE_RELATIONS
        ) else None
        return {
            "subject": path[0]["subject"],
            "object": path[-1]["object"],
            "relation": composed,
            "hops": n_hops,
            "inferred": True,
            "confidence": round(conf, 3),
            "path": [
                {"subject": e["subject"], "relation": e["relation"],
                 "object": e["object"], "confidence": e["confidence"],
                 "fact_id": e["fact_id"]}
                for e in path
            ],
        }
